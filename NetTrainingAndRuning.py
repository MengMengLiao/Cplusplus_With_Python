import numpy as np
from LayerOfConvPooling import LayerOfConvPooling
from LayerOfHidden import HiddenLayer
from LayerOfLogisticReg import LogisticRegression
import theano.tensor as T
import theano as th
import timeit
import downhill
import math
import cv2
import os
import cPickle

class BackSubConvNetwork(object):

    def __init__(self, learning_rate = 0.001, n_epoches = 15, nkerns = [6, 16], batch_size = 100):

        self.learningRate = learning_rate
        self.Nepoches     = n_epoches

        self.Nkerns       = nkerns
        self.BatchSize    = batch_size

        self.IsTestModelConstructed = False
        self.IsTestModelLoaded      = False

    def ExtractImagePath(self, ImagePath):

        IsFileNameEqual = True
        Index_i = 0
        LenOfFileName = 0
        ImageNameList = []
        ImageFileList = []

        for filename in os.listdir(ImagePath):

            if LenOfFileName == 0:
                LenOfFileName = len(filename)

            if len(filename) != LenOfFileName:
                IsFileNameEqual = False
                break

            if (Index_i + 1) % 10 == 0:
                break

            Index_i += 1

        if IsFileNameEqual:

            for filename in os.listdir(ImagePath):
                ImageNameList.append(filename)

            for Index_i in range(len(ImageNameList) - 1):
                for Index_j in range(len(ImageNameList) - Index_i - 1):

                    if ImageNameList[Index_j] > ImageNameList[Index_j + 1]:
                        temp = ImageNameList[Index_j]
                        ImageNameList[Index_j] = ImageNameList[Index_j + 1]
                        ImageNameList[Index_j + 1] = temp

            for Index_i in range(len(ImageNameList)):

                ImageFileList.append(os.path.join(ImagePath, ImageNameList[Index_i]))
        else:

            for filename in os.listdir(ImagePath):
                ImageNameList.append(filename)

            ImageIndexNum = []

            for Index_i in range(len(ImageNameList)):
                ImageIndexNum.append(ImageNameList[Index_i].split('.')[0])

            TempStrOne = ""
            TempStrTwo = ""
            IsStartIndex = False
            Index_Start_one = 1
            Index_Start_two = 1
            Index_End_one = 0
            Index_End_tow = 0
            while TempStrOne == "" and TempStrTwo == "":

                tempStart = Index_End_one
                IsStartIndex = False
                for Index_i in range(tempStart, len(ImageIndexNum[0])):

                    if ImageIndexNum[0][Index_i] >= '0' and ImageIndexNum[0][Index_i] <= '9':

                        if IsStartIndex == False:
                            Index_Start_one = Index_i

                        Index_End_one = Index_i
                        TempStrOne += ImageIndexNum[0][Index_i]
                        IsStartIndex = True

                    elif IsStartIndex:
                        break

                IsStartIndex = False
                tempStart = Index_End_tow
                for Index_i in range(tempStart, len(ImageIndexNum[1])):

                    if ImageIndexNum[1][Index_i] >= '0' and ImageIndexNum[1][Index_i] <= '9':

                        if IsStartIndex == False:
                            Index_Start_two = Index_i

                        Index_End_tow = Index_i

                        TempStrTwo += ImageIndexNum[1][Index_i]
                        IsStartIndex = True

                    elif IsStartIndex:
                        break

                if int(TempStrOne) != int(TempStrTwo):

                    if Index_Start_one == Index_Start_two:

                        break

                    else:

                        print ('Image file name is unsuitable for reading.')
                        break
                else:

                    TempStrOne = ""
                    TempStrTwo = ""

                    if Index_Start_one == Index_End_one:
                        print('Image file name is unsuitable for reading')
                        break

            for Index_i in range(len(ImageIndexNum)):

                TempIndexStr = ""
                for Index_j in range(Index_Start_one, len(ImageIndexNum[Index_i])):

                    if ImageIndexNum[Index_i][Index_j] >= '0' and ImageIndexNum[Index_i][Index_j] <= '9':
                        TempIndexStr += ImageIndexNum[Index_i][Index_j]

                ImageIndexNum[Index_i] = int(TempIndexStr)

            for Index_i in range(len(ImageIndexNum) - 1):

                for Index_j in range(len(ImageIndexNum) - 1 - Index_i):

                    if (ImageIndexNum[Index_j] > ImageIndexNum[Index_j + 1]):
                        temp = ImageIndexNum[Index_j]
                        ImageIndexNum[Index_j] = ImageIndexNum[Index_j + 1]
                        ImageIndexNum[Index_j + 1] = temp

                        temp = ImageNameList[Index_j]
                        ImageNameList[Index_j] = ImageNameList[Index_j + 1]
                        ImageNameList[Index_j + 1] = temp

            for Index_i in range(len(ImageNameList)):
                ImageFileList.append(os.path.join(ImagePath, ImageNameList[Index_i]))

        return ImageFileList

    def GenerateTrainningDatasAndLables(self, BkgImg, TrainImg, PatchSize,  LableImg=None, ROICoordinate=None, NumOfBatchContains=100):


        Debug = False
        # Adding External Padding on the Image
        PaddingBkgImg   = np.zeros((BkgImg.shape[0]   + ((PatchSize // 2) * 2), BkgImg.shape[1]   + ((PatchSize // 2) * 2)))
        PaddingTrainImg = np.zeros((TrainImg.shape[0] + ((PatchSize // 2) * 2), TrainImg.shape[1] + ((PatchSize // 2) * 2)))

        # Inserting the processed image into the padding image
        PaddingBkgImg  [(PatchSize // 2): (PatchSize // 2 + BkgImg.shape[0]  ), (PatchSize // 2): (PatchSize // 2 + BkgImg.shape[1])]   = BkgImg
        PaddingTrainImg[(PatchSize // 2): (PatchSize // 2 + TrainImg.shape[0]), (PatchSize // 2): (PatchSize // 2 + TrainImg.shape[1])] = TrainImg

        if LableImg is not None:

            if Debug:
                cv2.imshow("Initial Label Image", LableImg)

            train_set_y = LableImg.reshape((LableImg.shape[0] * LableImg.shape[1], 1))

            NumOfPosLable = 0
            NumOfPosList  = []
            NumOfNegLable = 0
            NumOfNegList  = []

            IsExistPos = False
            PatchIndex = 0
            for i in range(train_set_y.shape[0]):

                if train_set_y[i] == 1:
                    IsExistPos = True

                if (i + 1) % NumOfBatchContains == 0:

                    if IsExistPos:

                        NumOfPosLable += 1
                        NumOfPosList.append(PatchIndex)
                    else:

                        NumOfNegLable += 1
                        NumOfNegList.append(PatchIndex)

                    PatchIndex += 1

                    IsExistPos = False
            # Do the judgement about weather contains positive pixel in each batch. Each batch contains 100/200/300... pixels
            # NumofPosList store the index of the batch which does not contain positive pixel
            # NumofNegList store the index of the batch which contains positive pixel


            ReConstructList = []
            if NumOfNegLable > NumOfPosLable:

                NumOfSplitInsert = math.ceil(NumOfNegLable / float(NumOfPosLable))
                NumOfSplitInsert = int(NumOfSplitInsert)

                IndexOfPosList = 0

                for i in range(NumOfNegLable):

                    ReConstructList.append(NumOfNegList[i])
                    if (i + 1) % NumOfSplitInsert == 0:
                        ReConstructList.append(NumOfPosList[IndexOfPosList])
                        IndexOfPosList += 1

                while IndexOfPosList < len(NumOfPosList):

                    RandIndex = np.random.randint(0, len(ReConstructList))
                    ReConstructList.insert(RandIndex, NumOfPosList[IndexOfPosList])
                    IndexOfPosList += 1

            else:

                NumOfSplitInsert = math.ceil(NumOfPosLable / float(NumOfNegLable))
                NumOfSplitInsert = int(NumOfSplitInsert)

                IndexOfNegList = 0

                for i in range(NumOfPosLable):

                    ReConstructList.append(NumOfPosList[i])
                    if (i + 1) % NumOfSplitInsert == 0:
                        ReConstructList.append(NumOfNegList[IndexOfNegList])
                        IndexOfNegList += 1

                while IndexOfNegList < len(NumOfNegList):

                    RandIndex = np.random.randint(0, len(ReConstructList))
                    ReConstructList.insert(RandIndex, NumOfPosList[IndexOfNegList])
                    IndexOfNegList += 1

            #Mix the batches which contain positive pixels and no positive pixels. Reconstruct List stores the index of each batches.

            train_set_x  = np.zeros((BkgImg.shape[0] * BkgImg.shape[1], 2, PatchSize, PatchSize))
            MixTrainSetY = np.zeros((LableImg.shape[0] * LableImg.shape[1], 1))
            IndexOfTrainSet = 0

            for i in range(len(ReConstructList)):

                #Constructing the new label sets and storing in the MixTrainSetY
                for j in range(ReConstructList[i] * NumOfBatchContains, (ReConstructList[i] * NumOfBatchContains) + NumOfBatchContains):

                    MixTrainSetY[i * NumOfBatchContains + j - (ReConstructList[i] * NumOfBatchContains)] = train_set_y[j]


                col_start = (ReConstructList[i] * NumOfBatchContains) % BkgImg.shape[1];
                col_End   = (ReConstructList[i] * NumOfBatchContains + NumOfBatchContains - 1) % BkgImg.shape[1];

                if col_End > col_start:

                    row = (ReConstructList[i] * NumOfBatchContains) // BkgImg.shape[1];
                    for m in range(col_start, col_End + 1):

                        train_set_x[IndexOfTrainSet, 0] = PaddingTrainImg[row: row + PatchSize, m: m + PatchSize]
                        train_set_x[IndexOfTrainSet, 1] = PaddingBkgImg  [row: row + PatchSize, m: m + PatchSize]
                        IndexOfTrainSet += 1

                else:

                    rowOne = (ReConstructList[i] * NumOfBatchContains) // BkgImg.shape[1];
                    rowTwo = (ReConstructList[i] * NumOfBatchContains + NumOfBatchContains - 1) // BkgImg.shape[1];

                    for m in range(col_start, BkgImg.shape[1]):

                        train_set_x[IndexOfTrainSet, 0] = PaddingTrainImg[rowOne: rowOne + PatchSize, m: m + PatchSize]
                        train_set_x[IndexOfTrainSet, 1] = PaddingBkgImg  [rowOne: rowOne + PatchSize, m: m + PatchSize]
                        IndexOfTrainSet += 1

                    for m in range(0, col_End + 1):

                        train_set_x[IndexOfTrainSet, 0] = PaddingTrainImg[rowTwo: rowTwo + PatchSize, m: m + PatchSize]
                        train_set_x[IndexOfTrainSet, 1] = PaddingBkgImg  [rowTwo: rowTwo + PatchSize, m: m + PatchSize]
                        IndexOfTrainSet += 1
            # According to the mixed batch index to generate the real training data

            if Debug:

                NewLableImage = MixTrainSetY.reshape((LableImg.shape[0], LableImg.shape[1]))
                cv2.imshow("Mixed Label Image", NewLableImage)

            return [train_set_x, MixTrainSetY]
        else:

            if ROICoordinate is not None:
                Y = ROICoordinate[0]
                X = ROICoordinate[1]

                Height = ROICoordinate[2]
                Width  = ROICoordinate[3]

                train_set_x = np.zeros(((Width + 1) * (Height + 1), 2, PatchSize, PatchSize), dtype=th.config.floatX)
                for i in range(Y, Y + Height + 1):

                    for j in range(X, X + Width + 1):

                        train_set_x[(i - Y) * (Width + 1) + (j - X), 0] = PaddingTrainImg[i:(i + PatchSize), j:(j + PatchSize)]
                        train_set_x[(i - Y) * (Width + 1) + (j - X), 1] = PaddingBkgImg  [i:(i + PatchSize), j:(j + PatchSize)]

                return train_set_x

    def ConstructNetModel(self):

        self.rng = np.random.RandomState(23455)

        self.x = T.tensor4('Input')
        self.y = T.imatrix('Output')

        self.layer0_input = self.x

        self.layer0 = LayerOfConvPooling(

            rng  = self.rng,
            Input= self.layer0_input,
            image_shape  = (None, 2, 27, 27),
            filter_shape = (self.Nkerns[0], 2, 5, 5),
            poolsize = (3, 3)
        )

        self.layer1 = LayerOfConvPooling(

            rng  = self.rng,
            Input= self.layer0.out,
            image_shape  = (None, self.Nkerns[0], 9, 9),
            filter_shape = (self.Nkerns[1], self.Nkerns[0], 5, 5),
            poolsize = (3, 3)
        )

        self.layer2_input = self.layer1.out.flatten(2)

        self.layer2 = HiddenLayer(

            rng  = self.rng,
            Input= self.layer2_input,
            n_in = self.Nkerns[1] * 3 * 3,
            n_out= 120,
            activation = T.tanh
        )

        self.layer3 = LogisticRegression(

            rng   = self.rng,
            Input = self.layer2.output,
            n_in  = 120,
            n_out = 1
        )

    def BackSubNetTrainningByConstantLoops(self, TrainningImage, LableImage, BackgroundImage, ModelSavingFile):

        # Debug programming
        # TrainImageFile = open("TrainImageFile.txt", 'w')
        # for Index_i in range(TrainningImage.shape[0]):
        #
        #     for Index_j in range(TrainningImage.shape[1]):
        #
        #         if TrainningImage[Index_i][Index_j] < 10:
        #
        #             TrainImageFile.write(str(TrainningImage[Index_i][Index_j]) + "   ")
        #         elif TrainningImage[Index_i][Index_j] >= 10 and TrainningImage[Index_i][Index_j] < 100:
        #
        #             TrainImageFile.write(str(TrainningImage[Index_i][Index_j]) + "  ")
        #         else:
        #
        #             TrainImageFile.write(str(TrainningImage[Index_i][Index_j]) + " ")
        #
        #     TrainImageFile.write("\n")
        # TrainImageFile.close()
        #
        # TrainBkImageFile = open("TrainBKImageFile.txt", 'w')
        # for Index_i in range(BackgroundImage.shape[0]):
        #
        #     for Index_j  in range(BackgroundImage.shape[1]):
        #
        #         if BackgroundImage[Index_i][Index_j] < 10:
        #
        #             TrainBkImageFile.write(str(BackgroundImage[Index_i][Index_j]) + "   ")
        #         elif BackgroundImage[Index_i][Index_j] >= 10 and BackgroundImage[Index_i][Index_j] < 100:
        #
        #             TrainBkImageFile.write(str(BackgroundImage[Index_i][Index_j]) + "  ")
        #         else:
        #
        #             TrainBkImageFile.write(str(BackgroundImage[Index_i][Index_j]) + " ")
        #     TrainBkImageFile.write("\n")
        # TrainBkImageFile.close()

        print("Enter into training")
        LableImg = np.array(LableImage, dtype='int32')
        TrainImg = np.array(TrainningImage, dtype=th.config.floatX) / 255.0
        self.BkgImg = np.array(BackgroundImage, dtype=th.config.floatX) / 255.0

        DataSets_ScaleOne = self.GenerateTrainningDatasAndLables(self.BkgImg, TrainImg, 27, LableImg)

        ORGTrainSetX = np.asarray(DataSets_ScaleOne[0], dtype=th.config.floatX)
        ORGTrainSetY = np.asarray(DataSets_ScaleOne[1], dtype='int32')

        if DataSets_ScaleOne[0].shape[0] % self.BatchSize == 0:

            ORGTrainBatches = DataSets_ScaleOne[0].shape[0] // self.BatchSize
        else:

            ORGTrainBatches = DataSets_ScaleOne[0].shape[0] // self.BatchSize + 1

        # scaleTwo = 0.75

        RescaleTrainImg_One = cv2.resize(TrainImg, (int(TrainImg.shape[1] * 0.75), int(TrainImg.shape[0] * 0.75)), interpolation=cv2.INTER_NEAREST)
        RescaleBkgImg_One   = cv2.resize(self.BkgImg, (int(self.BkgImg.shape[1] * 0.75), int(self.BkgImg.shape[0] * 0.75)), interpolation=cv2.INTER_NEAREST)
        RescaleLableImg_One = cv2.resize(LableImg, (int(LableImg.shape[1] * 0.75), int(LableImg.shape[0] * 0.75)), interpolation=cv2.INTER_NEAREST)

        DataSets_ScaleTwo = self.GenerateTrainningDatasAndLables(RescaleBkgImg_One, RescaleTrainImg_One, 27, RescaleLableImg_One)

        RSCTrainSetX_One = np.asarray(DataSets_ScaleTwo[0], dtype=th.config.floatX)
        RSCTrainSetY_One = np.asarray(DataSets_ScaleTwo[1], dtype=th.config.floatX)

        Batches_One = int(math.ceil(DataSets_ScaleTwo[0].shape[0] / float(ORGTrainBatches)))
        RSCTrainBatches_One = int(math.ceil(DataSets_ScaleTwo[0].shape[0] / float(Batches_One)))

        # scaleThree = 0.5

        RescaleTrainImg_Two = cv2.resize(TrainImg, (int(TrainImg.shape[1] * 0.50), int(TrainImg.shape[0] * 0.50)), interpolation=cv2.INTER_NEAREST)
        RescaleBkgImg_Two   = cv2.resize(self.BkgImg, (int(self.BkgImg.shape[1] * 0.50), int(self.BkgImg.shape[0] * 0.50)), interpolation=cv2.INTER_NEAREST)
        RescaleLableImg_Two = cv2.resize(LableImg, (int(LableImg.shape[1] * 0.50), int(LableImg.shape[0] * 0.50)), interpolation=cv2.INTER_NEAREST)

        DataSets_ScaleThree = self.GenerateTrainningDatasAndLables(RescaleBkgImg_Two, RescaleTrainImg_Two, 27, RescaleLableImg_Two)

        RSCTrainSetX_Two = np.asarray(DataSets_ScaleThree[0], dtype=th.config.floatX)
        RSCTrainSetY_Two = np.asarray(DataSets_ScaleThree[1], dtype=th.config.floatX)

        Batches_Two = int(math.ceil(DataSets_ScaleThree[0].shape[0] / float(ORGTrainBatches)))
        RSCTrainBatches_Two = int(math.ceil(DataSets_ScaleThree[0].shape[0] / float(Batches_Two)))

        print 'DefaultBatch = ', self.BatchSize, ', Batches_One = ', Batches_One, ', Batches_Two = ', Batches_Two
        print 'ORGTrainBatches = ', ORGTrainBatches, ', RSCTrainBatches_One = ', RSCTrainBatches_One, ',RSCTrainBatches_Two = ', RSCTrainBatches_Two

        cost = self.layer3.Cross_entropyErrorFunction(self.y)
        updates = downhill.build('rmsprop', loss=cost).get_updates(batch_size=self.BatchSize, learning_rate=self.learningRate)

        # params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        #
        # grads = T.grad(cost, params)
        #
        # updates = [
        #     (params_i, params_i - self.learningRate * grads_i)
        #     for params_i, grads_i in zip(params, grads)
        # ]

        print('... constructing the model ...')
        train_model = th.function(

            [self.x, self.y],
            cost,
            updates=list(updates)
        )

        print('... training ...')
        start_time = timeit.default_timer()

        epoch = 0
        TimesOfWrongClassified = 0
        LastTrainningLoss = 0
        while epoch < self.Nepoches or TimesOfWrongClassified < 5:
            loop_start = timeit.default_timer()
            epoch = epoch + 1
            print "loop " + str(epoch) + " :"
            OrgAverageLoss    = 0
            RSCAverageLossOne = 0
            RSCAverageLossTwo = 0
            TotalAverageLoss  = 0

            ScaleOneIndex = 0
            ScaleTwoIndex = 0

            for minibatch_index in range(ORGTrainBatches):

                #Original
                if (minibatch_index + 1) * self.BatchSize <= ORGTrainSetX.shape[0]:
                    Org_input = np.zeros((self.BatchSize, 2, 27, 27), dtype=th.config.floatX)
                else:
                    Org_input = np.zeros(((ORGTrainSetX.shape[0] - minibatch_index * self.BatchSize), 2, 27, 27), dtype=th.config.floatX)

                Org_input[:, 0] = ORGTrainSetX[minibatch_index * self.BatchSize: (minibatch_index + 1) * self.BatchSize, 0]
                Org_input[:, 1] = ORGTrainSetX[minibatch_index * self.BatchSize: (minibatch_index + 1) * self.BatchSize, 1]

                if (minibatch_index + 1) * self.BatchSize <= ORGTrainSetY.shape[0]:
                    Org_Lable = np.zeros((self.BatchSize, 1), dtype='int32')
                else:
                    Org_Lable = np.zeros(((ORGTrainSetY.shape[0] - (minibatch_index * self.BatchSize)), 1), dtype='int32')

                Org_Lable[:] = ORGTrainSetY[minibatch_index * self.BatchSize: (minibatch_index + 1) * self.BatchSize]
                OrgCostValue = train_model(Org_input, Org_Lable)

                OrgAverageLoss += OrgCostValue

                #Rescale 0.75
                if ScaleOneIndex < RSCTrainBatches_One:

                    if (ScaleOneIndex + 1) * Batches_One <= RSCTrainSetX_One.shape[0]:
                        ScaleOne_input = np.zeros((Batches_One, 2, 27, 27), dtype=th.config.floatX)
                    else:
                        ScaleOne_input = np.zeros(((RSCTrainSetX_One.shape[0] - (ScaleOneIndex * Batches_One)), 2, 27, 27), dtype=th.config.floatX)

                    ScaleOne_input[:, 0] = RSCTrainSetX_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One, 0]
                    ScaleOne_input[:, 1] = RSCTrainSetX_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One, 1]

                    if (ScaleOneIndex + 1) * Batches_One <= RSCTrainSetY_One.shape[0]:
                        ScaleOne_Label = np.zeros((Batches_One, 1), dtype='int32')
                    else:
                        ScaleOne_Label = np.zeros(((RSCTrainSetY_One.shape[0] - ScaleOneIndex * Batches_One), 1), dtype='int32')

                    ScaleOne_Label[:] = RSCTrainSetY_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One]
                    RescaleCostValueOne = train_model(ScaleOne_input, ScaleOne_Label)

                    ScaleOneIndex += 1
                    RSCAverageLossOne += RescaleCostValueOne

                #Rescale 0.5
                if ScaleTwoIndex < RSCTrainBatches_Two:

                    if (ScaleTwoIndex + 1) * Batches_Two <= RSCTrainSetX_Two.shape[0]:
                        ScaleTwo_input = np.zeros((Batches_Two, 2, 27, 27), dtype=th.config.floatX)
                    else:
                        ScaleTwo_input = np.zeros(((RSCTrainSetX_Two.shape[0] - ScaleTwoIndex * Batches_Two), 2, 27, 27), dtype=th.config.floatX)

                    ScaleTwo_input[:, 0] = RSCTrainSetX_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two, 0]
                    ScaleTwo_input[:, 1] = RSCTrainSetX_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two, 1]

                    if (ScaleTwoIndex + 1) * Batches_Two <= RSCTrainSetY_Two.shape[0]:
                        ScaleTwo_Lable = np.zeros((Batches_Two, 1), dtype='int32')
                    else:
                        ScaleTwo_Lable = np.zeros(((RSCTrainSetY_Two.shape[0] - (ScaleTwoIndex* Batches_Two)), 1), dtype='int32')

                    ScaleTwo_Lable[:] = RSCTrainSetY_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two]
                    RescaleCostValueTwo = train_model(ScaleTwo_input, ScaleTwo_Lable)

                    ScaleTwoIndex += 1
                    RSCAverageLossTwo += RescaleCostValueTwo

            TotalAverageLoss = TotalAverageLoss + (OrgAverageLoss    / (minibatch_index + 1)) + (RSCAverageLossOne / (ScaleOneIndex )) + (RSCAverageLossTwo / (ScaleTwoIndex ))

            print ('OrgImage   Average Loss is:' + str(OrgAverageLoss    / (minibatch_index + 1)))
            print ('Rscale One Average Loss is:' + str(RSCAverageLossOne / (ScaleOneIndex )))
            print ('Rscale Two Average Loss is:' + str(RSCAverageLossTwo / (ScaleTwoIndex )))
            print ('Total      Average Loss is:' + str(TotalAverageLoss))
            loop_end = timeit.default_timer()

            if epoch > 1 and LastTrainningLoss < TotalAverageLoss:

                TimesOfWrongClassified += 1

            LastTrainningLoss = TotalAverageLoss

            print('Loop', epoch, 'costs %.2fm' %((loop_end - loop_start) / 60.))

        end_time = timeit.default_timer()
        print('Training Time is  %.2fm' % ((end_time - start_time) / 60.))
        print('TrainningFinished')

        # if ModelSavingFile != "None":
        #     TrainedModel = open(ModelSavingFile, 'wb')
        #     cPickle.dump(self.layer3.W.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer3.b.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer2.W.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer2.b.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer1.W.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer1.b.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer0.W.get_value(borrow=True), TrainedModel, -1)
        #     cPickle.dump(self.layer0.b.get_value(borrow=True), TrainedModel, -1)
        #     TrainedModel.close()

    def BackSubNetTrainningByValidSetStopping(self, TrainningImage, LableImage, BackgroundImage, ModelSavingFile):

        LableImg    = np.array(LableImage, dtype='int32')

        TrainImg    = np.array(TrainningImage , dtype=th.config.floatX) / 255.0
        self.BkgImg = np.array(BackgroundImage, dtype=th.config.floatX) / 255.0

        DataSets_ScaleOne = self.GenerateTrainningDatasAndLables(self.BkgImg, TrainImg, 27, LableImg)

        if DataSets_ScaleOne[0].shape[0] % self.BatchSize == 0:

            ORGTrainBatches = DataSets_ScaleOne[0].shape[0] // self.BatchSize
        else:

            ORGTrainBatches = DataSets_ScaleOne[0].shape[0] // self.BatchSize + 1

        SizeOfTrainSetX    = int(math.ceil(ORGTrainBatches * 0.8))

        ORGTrainSetX = np.asarray(DataSets_ScaleOne[0][0 : SizeOfTrainSetX * self.BatchSize], dtype=th.config.floatX)
        ORGTrainSetY = np.asarray(DataSets_ScaleOne[1][0 : SizeOfTrainSetX * self.BatchSize], dtype='int32')

        ORGValiadateSetX = np.asarray(DataSets_ScaleOne[0][SizeOfTrainSetX * self.BatchSize: ], dtype=th.config.floatX)
        ORGValiadateSetY = np.asarray(DataSets_ScaleOne[1][SizeOfTrainSetX * self.BatchSize: ], dtype='int32')

        # scaleTwo = 0.75

        RescaleTrainImg_One = cv2.resize(TrainImg, (int(TrainImg.shape[1] * 0.75), int(TrainImg.shape[0] * 0.75)), interpolation=cv2.INTER_NEAREST)
        RescaleBkgImg_One   = cv2.resize(self.BkgImg, (int(self.BkgImg.shape[1] * 0.75), int(self.BkgImg.shape[0] * 0.75)), interpolation=cv2.INTER_NEAREST)
        RescaleLableImg_One = cv2.resize(LableImg, (int(LableImg.shape[1] * 0.75), int(LableImg.shape[0] * 0.75)), interpolation=cv2.INTER_NEAREST)

        DataSets_ScaleTwo = self.GenerateTrainningDatasAndLables(RescaleBkgImg_One, RescaleTrainImg_One, 27, RescaleLableImg_One)


        Batches_One = int(math.ceil(DataSets_ScaleTwo[0].shape[0] / float(ORGTrainBatches)))
        RSCTrainBatches_One = int(math.ceil(DataSets_ScaleTwo[0].shape[0] / float(Batches_One)))

        SizeOfTrainSetX_One = int(math.ceil(RSCTrainBatches_One * 0.8))

        RSCTrainSetX_One = np.asarray(DataSets_ScaleTwo[0][0 : SizeOfTrainSetX_One * Batches_One], dtype=th.config.floatX)
        RSCTrainSetY_One = np.asarray(DataSets_ScaleTwo[1][0 : SizeOfTrainSetX_One * Batches_One], dtype='int32')

        RSCValiadateSetX_One = np.asarray(DataSets_ScaleTwo[0][SizeOfTrainSetX_One * Batches_One: ], dtype=th.config.floatX)
        RSCValiadateSetY_One = np.asarray(DataSets_ScaleTwo[1][SizeOfTrainSetX_One * Batches_One: ], dtype='int32')


        # scaleThree = 0.5

        RescaleTrainImg_Two = cv2.resize(TrainImg, (int(TrainImg.shape[1] * 0.50), int(TrainImg.shape[0] * 0.50)), interpolation=cv2.INTER_NEAREST)
        RescaleBkgImg_Two   = cv2.resize(self.BkgImg,(int(self.BkgImg.shape[1] * 0.50), int(self.BkgImg.shape[0] * 0.50)), interpolation=cv2.INTER_NEAREST)
        RescaleLableImg_Two = cv2.resize(LableImg, (int(LableImg.shape[1] * 0.50), int(LableImg.shape[0] * 0.50)), interpolation=cv2.INTER_NEAREST)

        DataSets_ScaleThree = self.GenerateTrainningDatasAndLables(RescaleBkgImg_Two, RescaleTrainImg_Two, 27, RescaleLableImg_Two)

        Batches_Two = int(math.ceil(DataSets_ScaleThree[0].shape[0] / float(ORGTrainBatches)))
        RSCTrainBatches_Two = int(math.ceil(DataSets_ScaleThree[0].shape[0] / float(Batches_Two)))

        SizeOfTrainSetX_Two = int(math.ceil(RSCTrainBatches_Two * 0.8))

        RSCTrainSetX_Two = np.asarray(DataSets_ScaleThree[0][0 : SizeOfTrainSetX_Two * Batches_Two], dtype=th.config.floatX)
        RSCTrainSetY_Two = np.asarray(DataSets_ScaleThree[1][0 : SizeOfTrainSetX_Two * Batches_Two], dtype='int32')

        RSCValiadateSetX_Two = np.asarray(DataSets_ScaleThree[0][SizeOfTrainSetX_Two * Batches_Two: ], dtype=th.config.floatX)
        RSCValiadateSetY_Two = np.asarray(DataSets_ScaleThree[1][SizeOfTrainSetX_Two * Batches_Two: ], dtype='int32')

        print 'DefaultBatch = ', self.BatchSize, ', Batches_One = ', Batches_One, ', Batches_Two = ', Batches_Two
        print 'ORGTrainBatches = ', ORGTrainBatches, ', RSCTrainBatches_One = ', RSCTrainBatches_One, ',RSCTrainBatches_Two = ', RSCTrainBatches_Two
        print 'ORGTrainSetSize = ', SizeOfTrainSetX, ', SizeOfRSCTrainSetOne = ', SizeOfTrainSetX_One, ', SizeOfRSCTrainSet = ', SizeOfTrainSetX_Two

        cost    = self.layer3.Cross_entropyErrorFunction(self.y)
        updates = downhill.build('rmsprop', loss=cost).get_updates(batch_size = self.BatchSize, learning_rate = self.learningRate)

        train_model = th.function(

            [self.x, self.y],
            cost,
            updates = list(updates)
        )

        valid_model = th.function(

            [self.x, self.y],
            cost
        )

        print ('--------Start to Training--------')

        start_time = timeit.default_timer()

        epoch = 0
        StopTraining = 0
        TimesOfValidJump = 0
        LastAverValidLoss = 0
        while True:

            print "loop " + str(epoch + 1) + " :"

            loop_start = timeit.default_timer()

            OrgAverageLoss    = 0
            RSCAverageLossOne = 0
            RSCAverageLossTwo = 0
            TotalAverageLoss  = 0

            ScaleOneIndex = 0
            ScaleTwoIndex = 0

            # Training phase
            for BatchIndex in range(SizeOfTrainSetX):

                if (BatchIndex + 1) * self.BatchSize <= ORGTrainSetX.shape[0]:

                    Org_input = np.zeros((self.BatchSize, 2, 27, 27), dtype=th.config.floatX)
                else:

                    Org_input = np.zeros((ORGTrainSetX.shape[0] - BatchIndex * self.BatchSize, 2, 27,27), dtype=th.config.floatX)

                Org_input[:, 0] = ORGTrainSetX[BatchIndex * self.BatchSize: (BatchIndex + 1) * self.BatchSize, 0]
                Org_input[:, 1] = ORGTrainSetX[BatchIndex * self.BatchSize: (BatchIndex + 1) * self.BatchSize, 1]

                if (BatchIndex + 1) * self.BatchSize <= ORGTrainSetY.shape[0]:

                    Org_Lable = np.zeros((self.BatchSize, 1), dtype='int32')
                else:

                    Org_Lable = np.zeros((ORGTrainSetY.shape[0] - BatchIndex * self.BatchSize, 1), dtype='int32')

                Org_Lable[:] = ORGTrainSetY[BatchIndex * self.BatchSize : (BatchIndex + 1) * self.BatchSize]

                orgCostValue = train_model(Org_input, Org_Lable)
                OrgAverageLoss += orgCostValue

                if ScaleOneIndex < SizeOfTrainSetX_One:

                    if (ScaleOneIndex + 1) * Batches_One <= RSCTrainSetX_One.shape[0]:
                        ScaleOne_input = np.zeros((Batches_One, 2, 27, 27), dtype=th.config.floatX)
                    else:
                        ScaleOne_input = np.zeros((RSCTrainSetX_One.shape[0] - (ScaleOneIndex * Batches_One), 2, 27, 27), dtype=th.config.floatX)

                    ScaleOne_input[:, 0] = RSCTrainSetX_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One, 0]
                    ScaleOne_input[:, 1] = RSCTrainSetX_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One, 1]

                    if (ScaleOneIndex + 1) * Batches_One <= RSCTrainSetY_One.shape[0]:

                        ScaleOne_Label = np.zeros((Batches_One, 1), dtype='int32')
                    else:

                        ScaleOne_Label = np.zeros(((RSCTrainSetY_One.shape[0] - ScaleOneIndex * Batches_One), 1), dtype='int32')

                    ScaleOne_Label[:] = RSCTrainSetY_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One]
                    RescaleCostValueOne = train_model(ScaleOne_input, ScaleOne_Label)

                    ScaleOneIndex += 1
                    RSCAverageLossOne += RescaleCostValueOne

                if ScaleTwoIndex < SizeOfTrainSetX_Two:

                    if (ScaleTwoIndex + 1) * Batches_Two <= RSCTrainSetX_Two.shape[0]:
                        ScaleTwo_input = np.zeros((Batches_Two, 2, 27, 27), dtype=th.config.floatX)
                    else:
                        ScaleTwo_input = np.zeros((RSCTrainSetX_Two.shape[0] - ScaleTwoIndex * Batches_Two, 2, 27, 27), dtype=th.config.floatX)

                    ScaleTwo_input[:, 0] = RSCTrainSetX_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two, 0]
                    ScaleTwo_input[:, 1] = RSCTrainSetX_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two, 1]

                    if (ScaleTwoIndex + 1) * Batches_Two <= RSCTrainSetY_Two.shape[0]:
                        ScaleTwo_Lable = np.zeros((Batches_Two, 1), dtype='int32')
                    else:
                        ScaleTwo_Lable = np.zeros(((RSCTrainSetY_Two.shape[0] - (ScaleTwoIndex * Batches_Two)), 1), dtype='int32')

                    ScaleTwo_Lable[:] = RSCTrainSetY_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two]
                    RescaleCostValueTwo = train_model(ScaleTwo_input, ScaleTwo_Lable)

                    ScaleTwoIndex += 1
                    RSCAverageLossTwo += RescaleCostValueTwo

            TotalAverageLoss = TotalAverageLoss + (OrgAverageLoss / (BatchIndex + 1)) + (RSCAverageLossOne / (ScaleOneIndex)) + (RSCAverageLossTwo / (ScaleTwoIndex))

            print ('OrgImage   Average Loss is:' + str(OrgAverageLoss    / (BatchIndex + 1)))
            print ('Rscale One Average Loss is:' + str(RSCAverageLossOne / (ScaleOneIndex)))
            print ('Rscale Two Average Loss is:' + str(RSCAverageLossTwo / (ScaleTwoIndex)))
            print ('Total      Average Loss is:' + str(TotalAverageLoss))

            #Validate Phase

            OrgAverValidLoss    = 0
            RSCAverValidLossOne = 0
            RSCAverValidLossTwo = 0
            TotalAverValidLoss  = 0

            ScaleOneIndex = 0
            ScaleTwoIndex = 0
            for BatchIndex in range(0, ORGTrainBatches - SizeOfTrainSetX):

                if (BatchIndex + 1) * self.BatchSize <= ORGValiadateSetX.shape[0]:

                    OrgValid_Input = np.zeros((self.BatchSize, 2, 27, 27), dtype=th.config.floatX)
                else:

                    OrgValid_Input = np.zeros((ORGValiadateSetX.shape[0] - BatchIndex * self.BatchSize, 2, 27, 27), dtype=th.config.floatX)

                OrgValid_Input[:, 0] = ORGValiadateSetX[BatchIndex * self.BatchSize: (BatchIndex + 1) * self.BatchSize, 0]
                OrgValid_Input[:, 1] = ORGValiadateSetX[BatchIndex * self.BatchSize: (BatchIndex + 1) * self.BatchSize, 1]

                if (BatchIndex + 1) * self.BatchSize <= ORGValiadateSetY.shape[0]:

                    OrgValid_Lable = np.zeros((self.BatchSize, 1), dtype='int32')
                else:

                    OrgValid_Lable = np.zeros((ORGValiadateSetY.shape[0] - BatchIndex * self.BatchSize, 1), dtype='int32')

                OrgValid_Lable[:] = ORGValiadateSetY[BatchIndex * self.BatchSize : (BatchIndex + 1) * self.BatchSize]

                OrgValidLoss = valid_model(OrgValid_Input, OrgValid_Lable)
                OrgAverValidLoss  += OrgValidLoss

                if ScaleOneIndex < RSCTrainBatches_One - SizeOfTrainSetX_One:

                    if (ScaleOneIndex + 1) * Batches_One <= RSCValiadateSetX_One.shape[0]:
                        ScaleOne_input = np.zeros((Batches_One, 2, 27, 27), dtype=th.config.floatX)
                    else:
                        ScaleOne_input = np.zeros((RSCValiadateSetX_One.shape[0] - (ScaleOneIndex * Batches_One), 2, 27, 27), dtype=th.config.floatX)

                    ScaleOne_input[:, 0] = RSCValiadateSetX_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One, 0]
                    ScaleOne_input[:, 1] = RSCValiadateSetX_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One, 1]

                    if (ScaleOneIndex + 1) * Batches_One <= RSCValiadateSetY_One.shape[0]:
                        ScaleOne_Label = np.zeros((Batches_One, 1), dtype='int32')
                    else:
                        ScaleOne_Label = np.zeros((RSCValiadateSetY_One.shape[0] - ScaleOneIndex * Batches_One, 1), dtype='int32')

                    ScaleOne_Label[:] = RSCValiadateSetY_One[ScaleOneIndex * Batches_One: (ScaleOneIndex + 1) * Batches_One]
                    RSCOneValidLoss = valid_model(ScaleOne_input, ScaleOne_Label)

                    ScaleOneIndex += 1
                    RSCAverValidLossOne += RSCOneValidLoss

                if ScaleTwoIndex < RSCTrainBatches_Two - SizeOfTrainSetX_Two:

                    if (ScaleTwoIndex + 1) * Batches_Two <= RSCValiadateSetX_Two.shape[0]:

                        ScaleTwo_input = np.zeros((Batches_Two, 2, 27, 27), dtype=th.config.floatX)
                    else:

                        ScaleTwo_input = np.zeros(((RSCValiadateSetX_Two.shape[0] - ScaleTwoIndex * Batches_Two), 2, 27, 27), dtype=th.config.floatX)

                    ScaleTwo_input[:, 0] = RSCValiadateSetX_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two, 0]
                    ScaleTwo_input[:, 1] = RSCValiadateSetX_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two, 1]

                    if (ScaleTwoIndex + 1) * Batches_Two <= RSCValiadateSetY_Two.shape[0]:

                        ScaleTwo_Lable = np.zeros((Batches_Two, 1), dtype='int32')
                    else:

                        ScaleTwo_Lable = np.zeros((RSCValiadateSetY_Two.shape[0] - (ScaleTwoIndex * Batches_Two), 1), dtype='int32')

                    ScaleTwo_Lable[:] = RSCValiadateSetY_Two[ScaleTwoIndex * Batches_Two: (ScaleTwoIndex + 1) * Batches_Two]

                    RSCTwoValidLoss = valid_model(ScaleTwo_input, ScaleTwo_Lable)

                    ScaleTwoIndex += 1
                    RSCAverValidLossTwo += RSCTwoValidLoss

            TotalAverValidLoss = TotalAverValidLoss + (OrgAverValidLoss / (BatchIndex + 1)) + (RSCAverValidLossOne / (ScaleOneIndex)) + (RSCAverValidLossTwo / (ScaleTwoIndex))

            print('Org   Valid Loss is:', OrgAverValidLoss / (BatchIndex + 1))
            print('Rescale One Loss is:', RSCAverValidLossOne / ScaleOneIndex)
            print('Rescale Two Loss is:', RSCAverValidLossTwo / ScaleTwoIndex)


            print('Cur Total Aver Valid Loss is:', TotalAverValidLoss, ', Last Total Aver Valid Loss is:', LastAverValidLoss)

            if epoch >= 1:

                if TotalAverValidLoss > LastAverValidLoss:

                    if StopTraining == 0:
                        TimesOfValidJump += 1
                    else:
                        TimesOfValidJump = 0

                    StopTraining += 1
                else:

                    if StopTraining != 0:
                        TimesOfValidJump += 1
                    else:
                        TimesOfValidJump = 0

                    StopTraining = 0

            LastAverValidLoss = TotalAverValidLoss

            loop_end = timeit.default_timer()
            print('Loop', epoch + 1, 'costs %.2fm' % ((loop_end - loop_start) / 60.))

            if StopTraining >= 3 or TimesOfValidJump >= 5:

                break

            epoch += 1

        end_time = timeit.default_timer()
        print('Training Time is  %.2fm' % ((end_time - start_time) / 60.))
        print('TrainningFinished')

        TrainedModel = open(ModelSavingFile, 'wb')
        cPickle.dump(self.layer3.W.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer3.b.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer2.W.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer2.b.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer1.W.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer1.b.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer0.W.get_value(borrow=True), TrainedModel, -1)
        cPickle.dump(self.layer0.b.get_value(borrow=True), TrainedModel, -1)
        TrainedModel.close()

    def SetBackgroundImage(self, CurBKImage):

        self.BkgImg = np.array(CurBKImage, dtype=th.config.floatX) / 255.0

    def BackSubNetRunning(self, OrgImage, ROIImage):

        print('......Start to Running......')

        if self.IsTestModelConstructed == False:

            self.test_model = th.function([self.x], self.layer3.p_y_given_x)
            self.IsTestModelConstructed = True

        TestImg = np.array(OrgImage, dtype=th.config.floatX) / 255.0

        Scales = [1, 0.75, 0.5]
        MultiScaleResult = []

        for Index_k in range(len(Scales)):

            # 1.First step: Down Sampling
            RSCTestImg   = cv2.resize(TestImg, (int(TestImg.shape[1] * Scales[Index_k]), int(TestImg.shape[0] * Scales[Index_k])), interpolation=cv2.INTER_NEAREST)
            RSCBkImg     = cv2.resize(self.BkgImg, (int(self.BkgImg.shape[1] * Scales[Index_k]), int(self.BkgImg.shape[0] * Scales[Index_k])), interpolation=cv2.INTER_NEAREST)
            RSCROIImage  = cv2.resize(ROIImage, (int(ROIImage.shape[1] * Scales[Index_k]), int(ROIImage.shape[0] * Scales[Index_k])), interpolation=cv2.INTER_NEAREST)
            ROIRegion    = self.SearchROIRegionCoordinate(RSCROIImage)

            BinaryResult = np.zeros((RSCROIImage.shape[0], RSCROIImage.shape[1]), dtype='uint8')

            # print "Scale:", Scales[Index_k]
            # RSCROIImage_3d = np.zeros((RSCROIImage.shape[0], RSCROIImage.shape[1], 3), dtype='uint8')
            # for Index_i in range(RSCROIImage.shape[0]):
            #
            #     for Index_j in range(RSCROIImage.shape[1]):
            #         RSCROIImage_3d[Index_i][Index_j][0] = RSCROIImage[Index_i][Index_j]
            #         RSCROIImage_3d[Index_i][Index_j][1] = RSCROIImage[Index_i][Index_j]
            #         RSCROIImage_3d[Index_i][Index_j][2] = RSCROIImage[Index_i][Index_j]
            #
            # for Index_i in range(len(ROIRegion)):
            #     ROICoordinate = ROIRegion[Index_i]
            #     cv2.rectangle(RSCROIImage_3d, (ROICoordinate[1] - 5, ROICoordinate[0] - 5), (ROICoordinate[1] + ROICoordinate[3] + 5, ROICoordinate[0] + ROICoordinate[2] + 5), (0, 0, 255))
            #
            # cv2.imshow("Rescale ROI Region", RSCROIImage_3d)

            for Index_i in range(len(ROIRegion)):

                ROICoordinate = ROIRegion[Index_i]

                ImgRow    = ROICoordinate[0]
                ImgColumn = ROICoordinate[1]
                Test_set_x = self.GenerateTrainningDatasAndLables(RSCBkImg, RSCTestImg, 27, ROICoordinate=ROICoordinate)

                if Test_set_x.shape[0] % self.BatchSize == 0:

                    testingBatches = Test_set_x.shape[0] // self.BatchSize
                else:

                    testingBatches = Test_set_x.shape[0] // self.BatchSize + 1

                if testingBatches == 0:
                    testingBatches += 1

                for minibatch_index in range(testingBatches):

                    InputPatch = Test_set_x[minibatch_index * self.BatchSize: (minibatch_index + 1) * self.BatchSize]

                    ForePropabiltiy = self.test_model(InputPatch)

                    for j in range(len(ForePropabiltiy)):

                        if ImgColumn > (ROICoordinate[1] + ROICoordinate[3]):
                            ImgColumn = ROICoordinate[1]
                            ImgRow = ImgRow + 1

                        if ForePropabiltiy[j] > 0.6:
                            BinaryResult[ImgRow, ImgColumn] = 255
                        ImgColumn = ImgColumn + 1

            # BinaryResult_3d = np.zeros((BinaryResult.shape[0], BinaryResult.shape[1], 3), dtype='uint8')
            #
            # for Index_i in range(BinaryResult.shape[0]):
            #
            #     for Index_j in range(BinaryResult.shape[1]):
            #         BinaryResult_3d[Index_i][Index_j][0] = BinaryResult[Index_i][Index_j]
            #         BinaryResult_3d[Index_i][Index_j][1] = BinaryResult[Index_i][Index_j]
            #         BinaryResult_3d[Index_i][Index_j][2] = BinaryResult[Index_i][Index_j]
            #
            # for Index_i in range(len(ROIRegion)):
            #     ROICoordinate = ROIRegion[Index_i]
            #     cv2.rectangle(BinaryResult_3d, (ROICoordinate[1] - 5, ROICoordinate[0] - 5), (ROICoordinate[1] + ROICoordinate[3] + 5, ROICoordinate[0] + ROICoordinate[2] + 5), (0, 0, 255))

            # cv2.imshow("Redetection Rescale ROI Region", BinaryResult_3d)
            # cv2.waitKey(0)
            # cv2.destroyWindow("Redetection Rescale ROI Region")
            # cv2.destroyWindow("Rescale ROI Region")

            MultiScaleResult.append(BinaryResult)

        return MultiScaleResult

    def IsModelFileExisting(self, ModelFile):

        if os.path.isfile(ModelFile):

            return 1

        return 0

    def BackSubNetRunningByLoadingModel(self, OrgImage, ROIImage, BkImage, ModelFile):

        #Load Model for testing
        if self.IsTestModelLoaded == False:

            print('......Start to Load the Model......')
            TrainedModel = open(ModelFile, 'rb')
            self.layer3.W.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer3.b.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer2.W.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer2.b.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer1.W.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer1.b.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer0.W.set_value(cPickle.load(TrainedModel), borrow=True)
            self.layer0.b.set_value(cPickle.load(TrainedModel), borrow=True)
            TrainedModel.close()

            self.TestLoadModel = th.function([self.x], self.layer3.p_y_given_x)
            self.BkgImg = np.array(BkImage, dtype=th.config.floatX) / 255.0

            self.IsTestModelLoaded = True

        TestImg = np.array(OrgImage, dtype=th.config.floatX) / 255.0


        Scales = [1, 0.75, 0.5]
        MultiScaleResult = []

        for Index_k in range(len(Scales)):

            # 1.First step: Down Sampling
            RSCTestImg  = cv2.resize(TestImg, (int(TestImg.shape[1] * Scales[Index_k]), int(TestImg.shape[0] * Scales[Index_k])), interpolation=cv2.INTER_NEAREST)
            RSCBkImg    = cv2.resize(self.BkgImg, (int(self.BkgImg.shape[1] * Scales[Index_k]), int(self.BkgImg.shape[0] * Scales[Index_k])), interpolation=cv2.INTER_NEAREST)
            RSCROIImage = cv2.resize(ROIImage, (int(ROIImage.shape[1] * Scales[Index_k]), int(ROIImage.shape[0] * Scales[Index_k])), interpolation=cv2.INTER_NEAREST)

            ROIRegion    = self.SearchROIRegionCoordinate(RSCROIImage)
            BinaryResult = np.zeros((RSCTestImg.shape[0], RSCTestImg.shape[1]), dtype='uint8')

            print "Scale:", Scales[Index_k]
            RSCROIImage_3d  = np.zeros((RSCROIImage.shape[0], RSCROIImage.shape[1], 3), dtype='uint8')
            for Index_i in range(RSCROIImage.shape[0]):

                for Index_j in range(RSCROIImage.shape[1]):

                    RSCROIImage_3d[Index_i][Index_j][0] = RSCROIImage[Index_i][Index_j]
                    RSCROIImage_3d[Index_i][Index_j][1] = RSCROIImage[Index_i][Index_j]
                    RSCROIImage_3d[Index_i][Index_j][2] = RSCROIImage[Index_i][Index_j]

            for Index_i in range(len(ROIRegion)):

                ROICoordinate = ROIRegion[Index_i]
                cv2.rectangle(RSCROIImage_3d, (ROICoordinate[1] - 5, ROICoordinate[0] - 5), (ROICoordinate[1] + ROICoordinate[3] + 5, ROICoordinate[0] + ROICoordinate[2] + 5), (0, 0, 255))

            cv2.imshow("Rescale ROI Region", RSCROIImage_3d)

            # self.BatchSize = 10000
            for Index_i in range(len(ROIRegion)):

                ROICoordinate = ROIRegion[Index_i]

                ImgRow    = ROICoordinate[0]
                ImgColumn = ROICoordinate[1]

                print 'Size Of Testing Pixel:', (ROICoordinate[2] + 1) * (ROICoordinate[3] + 1)
                Test_set_x = self.GenerateTrainningDatasAndLables(RSCBkImg, RSCTestImg, 27, ROICoordinate=ROICoordinate)

                print 'Size Of Test_set_x:', Test_set_x.shape[0]

                if Test_set_x.shape[0] % self.BatchSize == 0:

                    testingBatches = Test_set_x.shape[0] // self.BatchSize
                else:

                    testingBatches = Test_set_x.shape[0] // self.BatchSize + 1

                if testingBatches == 0:
                    testingBatches += 1

                for minibatch_index in range(testingBatches):

                    InputPatch = Test_set_x[minibatch_index * self.BatchSize: (minibatch_index + 1) * self.BatchSize]

                    ForePropabiltiy = self.TestLoadModel(InputPatch)

                    for j in range(len(ForePropabiltiy)):

                        if ImgColumn > (ROICoordinate[1] + ROICoordinate[3]):
                            ImgColumn = ROICoordinate[1]
                            ImgRow = ImgRow + 1

                        if ForePropabiltiy[j] > 0.6:
                            BinaryResult[ImgRow, ImgColumn] = 255
                        ImgColumn = ImgColumn + 1
            BinaryResult_3d = np.zeros((BinaryResult.shape[0], BinaryResult.shape[1], 3), dtype='uint8')

            for Index_i in range(BinaryResult.shape[0]):

                for Index_j in range(BinaryResult.shape[1]):

                    BinaryResult_3d[Index_i][Index_j][0] = BinaryResult[Index_i][Index_j]
                    BinaryResult_3d[Index_i][Index_j][1] = BinaryResult[Index_i][Index_j]
                    BinaryResult_3d[Index_i][Index_j][2] = BinaryResult[Index_i][Index_j]

            for Index_i in range(len(ROIRegion)):
                ROICoordinate = ROIRegion[Index_i]
                cv2.rectangle(BinaryResult_3d, (ROICoordinate[1] - 5, ROICoordinate[0] - 5), (ROICoordinate[1] + ROICoordinate[3] + 5, ROICoordinate[0] + ROICoordinate[2] + 5), (0, 0, 255))

            cv2.imshow("Redetection Rescale ROI Region", BinaryResult_3d)
            cv2.waitKey(0)

            cv2.destroyWindow("Rescale ROI Region")
            cv2.destroyWindow("Redetection Rescale ROI Region")

            MultiScaleResult.append(BinaryResult)

        return MultiScaleResult

    def SearchROIRegionCoordinate(self, ROIImage):

        [Rows, Columns] = ROIImage.shape
        ROINumber = np.zeros((Rows, Columns), dtype=np.uint16)

        ROIRegion = []

        Vector_Coordinate = []
        Index_num = 1
        for Index_i in range(Rows):

            for Index_j in range(Columns):

                if ROIImage[Index_i, Index_j] == 255 and ROINumber[Index_i, Index_j] == 0:

                    min_x = Index_j
                    min_y = Index_i

                    max_x = Index_j
                    max_y = Index_i

                    Vector_Coordinate.append([Index_i, Index_j])
                    ROINumber[Index_i, Index_j] = Index_num
                    PixelNumOfROI = 1
                    while (len(Vector_Coordinate) > 0):

                        [C_y, C_x] = Vector_Coordinate.pop(0)

                        if C_y - 1 >= 0:

                            # left-up corner:
                            if C_x - 1 >= 0 and ROIImage[C_y - 1, C_x - 1] == 255 and ROINumber[C_y - 1, C_x - 1] == 0:

                                Vector_Coordinate.append([C_y - 1, C_x - 1])
                                ROINumber[C_y - 1, C_x - 1] = Index_num
                                PixelNumOfROI += 1

                                if C_x - 1 < min_x:
                                    min_x = C_x - 1

                            if ROIImage[C_y - 1, C_x] == 255 and ROINumber[C_y - 1, C_x] == 0:
                                Vector_Coordinate.append([C_y - 1, C_x])
                                ROINumber[C_y - 1, C_x] = Index_num
                                PixelNumOfROI += 1

                            if C_x + 1 < Columns and ROIImage[C_y - 1, C_x + 1] == 255 and ROINumber[
                                        C_y - 1, C_x + 1] == 0:

                                Vector_Coordinate.append([C_y - 1, C_x + 1])
                                ROINumber[C_y - 1, C_x + 1] = Index_num
                                PixelNumOfROI += 1

                                if C_x + 1 > max_x:
                                    max_x = C_x + 1

                        if C_x - 1 >= 0 and ROIImage[C_y, C_x - 1] == 255 and ROINumber[C_y, C_x - 1] == 0:

                            Vector_Coordinate.append([C_y, C_x - 1])
                            ROINumber[C_y, C_x - 1] = Index_num
                            PixelNumOfROI += 1

                            if C_x - 1 < min_x:
                                min_x = C_x - 1

                        if C_x + 1 < Columns and ROIImage[C_y, C_x + 1] == 255 and ROINumber[C_y, C_x + 1] == 0:

                            Vector_Coordinate.append([C_y, C_x + 1])
                            ROINumber[C_y, C_x + 1] = Index_num
                            PixelNumOfROI += 1

                            if C_x + 1 > max_x:
                                max_x = C_x + 1

                        if C_y + 1 < Rows:

                            # left-up corner:
                            if C_x - 1 >= 0 and ROIImage[C_y + 1, C_x - 1] == 255 and ROINumber[C_y + 1, C_x - 1] == 0:

                                Vector_Coordinate.append([C_y + 1, C_x - 1])
                                ROINumber[C_y + 1, C_x - 1] = Index_num
                                PixelNumOfROI += 1

                                if C_x - 1 < min_x:
                                    min_x = C_x - 1

                                if C_y + 1 > max_y:
                                    max_y = C_y + 1

                            if ROIImage[C_y + 1, C_x] == 255 and ROINumber[C_y + 1, C_x] == 0:
                                Vector_Coordinate.append([C_y + 1, C_x])
                                ROINumber[C_y + 1, C_x] = Index_num
                                PixelNumOfROI += 1

                                if C_y + 1 > max_y:
                                    max_y = C_y + 1

                            if C_x + 1 < Columns and ROIImage[C_y + 1, C_x + 1] == 255 and ROINumber[
                                        C_y + 1, C_x + 1] == 0:

                                Vector_Coordinate.append([C_y + 1, C_x + 1])
                                ROINumber[C_y + 1, C_x + 1] = Index_num
                                PixelNumOfROI += 1

                                if C_x + 1 > max_x:
                                    max_x = C_x + 1

                                if C_y + 1 > max_y:
                                    max_y = C_y + 1

                    # if PixelNumOfROI <= 10:
                    #
                    #     for Index_m in range(min_y, max_y + 1):
                    #
                    #         for Index_n in range(min_x, max_x + 1):
                    #
                    #             if ROINumber[Index_m, Index_n] == Index_num:
                    #                 ROIImage[Index_m, Index_n] = 0
                    #                 ROINumber[Index_m, Index_n] = 0
                    # else:

                    ROIRegion.append([min_y, min_x, max_y - min_y, max_x - min_x])
                    Index_num += 1

        if len(ROIRegion) > 0:

            return ROIRegion
        else:
            ROIRegion.append([0, 0, Rows - 1, Columns - 1])
            return ROIRegion

