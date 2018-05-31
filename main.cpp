#include <iostream>
#include <opencv.hpp>
#include <Python.h>
#include <string>
#include <vector>
#include <ctime>
#include <numpy/arrayobject.h>

using namespace std;
void init_numpy(){

    import_array();
}
void GenerateBackgroundImage(const vector<char *> ImgPath, const uint16_t NumOfBackImg, cv::Mat &BKImage);
float AchieveTheParameter(char *ParamStr);
int main(int argc, char **argv)
{
    if(argc != 9){

        cout<<"Usage: ./BKSubTrainingAndRunning -TF~ -BS~ -BE~ -SE~ OrgImagePath/ GTImagePath/ SaveImagePath/ ModelStoreName"<<endl;
        return -1;
    }

    int Index_i = 0, Index_j = 0, Index_m = 0, Index_n = 0;
    uint16_t TFImage    = (uint16_t)AchieveTheParameter(argv[1]);
    uint16_t BSImage    = (uint16_t)AchieveTheParameter(argv[2]);
    uint16_t BEImage    = (uint16_t)AchieveTheParameter(argv[3]);
    uint16_t SEImage    = (uint16_t)AchieveTheParameter(argv[4]);

    cout<<"TFImage = "<<TFImage<<", BSImage = "<<BSImage<<", BEImage = "<<BEImage <<", SEImage = "<<SEImage <<endl;

    char  *OrgImgPath     = argv[5];
    char  *GTImagePath    = argv[6];
    char  *SaveImgPath    = argv[7];
    char  *ModelStoreName = argv[8];


    /*Python Function was Called which used for Training*/
    PyObject *pModule = nullptr;
    PyObject *pDict   = nullptr;

    /*setting the python environments*/
    Py_SetPythonHome("/home/liaojian/anaconda2/envs/tensorflow");
    Py_Initialize();
    init_numpy();

    /*Python Documents*/
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/liaojian/Documents/Programming/PythonWorkSpace/ModuleForBSubConvLayer/')");

    /*Processing Files*/
    pModule = PyImport_ImportModule("NetTrainingAndRuning");

    if (pModule != nullptr){

        pDict = PyModule_GetDict(pModule);
    }else{

        cout<<"pModule Import Error"<<endl;
        PyErr_Print();
        return -1;
    }

    /*The Name of the Class*/
    PyObject *pClass = PyDict_GetItemString(pDict, "BackSubConvNetwork");
    if(pClass == nullptr){

        cout<<"pClass Loading Error"<<endl;
        PyErr_Print();
        return -1;
    }

    /*Initialization the class*/
    PyObject *pInstance = PyObject_CallObject (pClass, nullptr);
    if(pInstance == nullptr){

        cout<<"pInstance Initilization Error"<<endl;
        PyErr_Print();
        return -1;
    }

    /*Call the function of the class which used for constructing the network models*/
    PyObject_CallMethod(pInstance, "ConstructNetModel", nullptr);


    /*List of Image Path was generated*/
    PyObject *ImgPathList   = PyObject_CallMethod(pInstance, "ExtractImagePath", "(s)", OrgImgPath);
    PyObject *GTImgPathList = PyObject_CallMethod(pInstance, "ExtractImagePath", "(s)", GTImagePath);
    if(ImgPathList == nullptr || GTImgPathList == nullptr){

        cout<<"Image Path Generation Method is Called Failed!"<<endl;
        PyErr_Print();
        return -1;
    }

    /*Generate Bk Image*/
    cv::Mat BkImage;
    vector<char *> BkImagePath;
    for(Index_i = BSImage; Index_i <= BEImage; Index_i++){

        PyObject *ImgTPath = PyList_GetItem(ImgPathList, Index_i);
        BkImagePath.push_back(PyString_AsString(ImgTPath));
    }
    GenerateBackgroundImage(BkImagePath, BEImage - BSImage + 1, BkImage);

    /*Determine the model file exist or not*/
    PyObject *ModelFile = PyString_FromFormat(ModelStoreName);
    PyObject *ArgOfModelFile = PyTuple_New(1);
    PyTuple_SetItem(ArgOfModelFile, 0, ModelFile);

    PyObject *FuncNameOfModelFile = PyString_FromFormat("IsModelFileExisting");
    PyObject *FuncOfModelFile     = PyObject_GetAttr(pInstance, FuncNameOfModelFile);
    PyObject *ModelFileExisting   = nullptr;
    ModelFileExisting = PyObject_CallObject(FuncOfModelFile, ArgOfModelFile);

    if(ModelFileExisting == nullptr){

        cout<<"Model File Existing Function Calling Failed!"<<endl;
        PyErr_Print();
        return -1;
    }

    int IsModelFile = PyInt_AsLong(ModelFileExisting);

    if (IsModelFile == 0){

        // Training Phase
        cv::Mat TrainningImage, LableImage;

        PyObject *TIP_Ojbect = PyList_GetItem(ImgPathList,   TFImage);
        char *TrainImgPath   = PyString_AsString(TIP_Ojbect);

        PyObject *GIP_Object = PyList_GetItem(GTImgPathList, TFImage);
        char *GTImgPath      = PyString_AsString(GIP_Object);

        string TIP_Str(TrainImgPath);
        string GTI_Str(GTImgPath) ;

        TrainningImage = cv::imread(TIP_Str, CV_LOAD_IMAGE_GRAYSCALE);
        LableImage     = cv::imread(GTI_Str, CV_LOAD_IMAGE_GRAYSCALE);

        for(Index_i = 0; Index_i < LableImage.rows; Index_i++){

            uint8_t *LabImgPtr = LableImage.ptr<uint8_t>(Index_i);
            for(Index_j = 0; Index_j < LableImage.cols; Index_j++){

                if (LabImgPtr[Index_j] == 255){

                    LabImgPtr[Index_j] = LabImgPtr[Index_j] / 255;
                }else{

                    LabImgPtr[Index_j] = 0;
                }
            }
        }

        /*Parameter setting*/
        npy_intp Dim_TI[2] = {TrainningImage.rows, TrainningImage.cols};
        PyObject *ArrTrainImage = PyArray_SimpleNewFromData(2, Dim_TI, NPY_UINT8, TrainningImage.data);

        npy_intp Dim_LI[2] = {LableImage.rows, LableImage.cols};
        PyObject *ArrLableImage = PyArray_SimpleNewFromData(2, Dim_LI, NPY_UINT8, LableImage.data);

        npy_intp Dim_BI[2] = {BkImage.rows, BkImage.cols};
        PyObject *ArrBKImage    = PyArray_SimpleNewFromData(2, Dim_BI, NPY_UINT8, BkImage.data);

        PyObject *ArgArray_Training = PyTuple_New(4);
        PyTuple_SetItem(ArgArray_Training, 0, ArrTrainImage);
        PyTuple_SetItem(ArgArray_Training, 1, ArrLableImage);
        PyTuple_SetItem(ArgArray_Training, 2, ArrBKImage);

        PyObject *ModelSaving = PyString_FromFormat(ModelStoreName);
        PyTuple_SetItem(ArgArray_Training, 3, ModelSaving);

        /*Calling the training function*/
        PyObject *TrainFuncName = PyString_FromFormat("BackSubNetTrainning");
        PyObject *TrainFunc     = PyObject_GetAttr(pInstance, TrainFuncName);
        if(PyObject_CallObject(TrainFunc, ArgArray_Training) == nullptr){

            cout<<"Training Function Calling Error"<<endl;
            PyErr_Print();
            return -1;
        }

        Py_DECREF(ArrTrainImage);
        Py_DECREF(ArrLableImage);
        Py_DECREF(ArrBKImage);
        Py_DECREF(ModelSaving);
        Py_DECREF(ArgArray_Training);
        Py_DECREF(TrainFuncName);

        TrainningImage.release();
        LableImage.release();

    }

    char *SavingPath = new char[6 + strlen(SaveImgPath) + 2];
    sprintf(SavingPath, "mkdir %s", SaveImgPath);
    system(SavingPath);
    delete []SavingPath;

    for(Index_i = 0; Index_i < 4; Index_i++){

        char *SavingPath = new char[6 + strlen(SaveImgPath) + 11];
        if (Index_i == 3){

            sprintf(SavingPath, "mkdir %s/Result"  , SaveImgPath);
        }else{

            sprintf(SavingPath, "mkdir %s/Scale_%d", SaveImgPath, Index_i);
        }
        system (SavingPath);
        delete []SavingPath;
    }
    char ImgFullSavingPath[512];

    cv::Mat RunningImage, ROIImage, FinalSegResult, HolesSearching, Segmentations, ScaleImage, FinalResult;

    PyObject *RunFuncName = nullptr;
    PyObject *RunFunc     = nullptr;

    if(IsModelFile){

        RunFuncName = PyString_FromFormat("BackSubNetRunningByLoadingModel");
        RunFunc     = PyObject_GetAttr(pInstance, RunFuncName);
    }else{

        RunFuncName = PyString_FromFormat("BackSubNetRunning");
        RunFunc     = PyObject_GetAttr(pInstance, RunFuncName);
    }

    PyObject *Arr_ROIImg  = nullptr, *Arr_RunImg = nullptr, *Arr_BKImg = nullptr, *Arr_Model = nullptr, *ArgArray_Running = nullptr, *SegImgResult = nullptr;
    time_t ProcessStart,ProcessEnd;
    if(PyList_Check(ImgPathList)){

        int SizeOfList = PyList_Size(ImgPathList);
        for(Index_i = 0; Index_i < SizeOfList; Index_i++){


            Arr_RunImg       = nullptr;
            ArgArray_Running = nullptr;
            SegImgResult     = nullptr;
            Arr_ROIImg       = nullptr;
            Arr_BKImg        = nullptr;
            Arr_Model        = nullptr;

            PyObject *Item   = PyList_GetItem(ImgPathList, Index_i);

            cout<<"Started to Processing Image "<<Index_i<<" :"<<endl;

            string RunImgPath(PyString_AsString(Item));
            RunningImage = cv::imread(RunImgPath, CV_LOAD_IMAGE_GRAYSCALE);
            RunImgPath.clear();

            PyObject *ROIItem = PyList_GetItem(GTImgPathList, Index_i);
            string   ROIImgPath(PyString_AsString(ROIItem));
            ROIImage = cv::imread(ROIImgPath, CV_LOAD_IMAGE_GRAYSCALE);

            for(Index_m = 0; Index_m < ROIImage.rows; Index_m++){

                uint8_t *RIPtr = ROIImage.ptr<uint8_t>(Index_m);
                for(Index_n = 0; Index_n < ROIImage.cols; Index_n++){

                    if(RIPtr[Index_n] != 255){
                        RIPtr[Index_n] = 0;
                    }
                }
            }

            if (Index_i >= SEImage){

                npy_intp RunImgDims[2] = {RunningImage.rows, RunningImage.cols};
                Arr_RunImg = PyArray_SimpleNewFromData(2, RunImgDims, NPY_UINT8, RunningImage.data);

                if(Arr_RunImg == nullptr){

                    cout<<"Runing Image Store into the Memory Error."<<endl;
                    PyErr_Print();
                    return -1;
                }

                npy_intp ROIImg_Dims[2] = {RunningImage.rows, RunningImage.cols};
                Arr_ROIImg = PyArray_SimpleNewFromData(2, ROIImg_Dims, NPY_UINT8, ROIImage.data);

                if(Arr_ROIImg == nullptr){

                    cout<<"ROI Image Store into the Memory Error."<<endl;
                    PyErr_Print();
                    return -1;
                }

                npy_intp Dim_BI[2] = {BkImage.rows, BkImage.cols};
                Arr_BKImg  = PyArray_SimpleNewFromData(2, Dim_BI, NPY_UINT8, BkImage.data);

                if(Arr_BKImg == nullptr){

                    cout<<"Background Image Store into the Memory Error."<<endl;
                    PyErr_Print();
                    return -1;
                }

                Arr_Model = PyString_FromFormat(ModelStoreName);

                if (IsModelFile){

                    ArgArray_Running = PyTuple_New(4);
                    PyTuple_SetItem(ArgArray_Running, 0, Arr_RunImg);
                    PyTuple_SetItem(ArgArray_Running, 1, Arr_ROIImg);
                    PyTuple_SetItem(ArgArray_Running, 2, Arr_BKImg );
                    PyTuple_SetItem(ArgArray_Running, 3, Arr_Model );

                }else{

                    ArgArray_Running = PyTuple_New(2);
                    PyTuple_SetItem(ArgArray_Running, 0, Arr_RunImg);
                    PyTuple_SetItem(ArgArray_Running, 1, Arr_ROIImg);
                }

                cout<<"Input Image which used for Runing is prepared, Runing Started:"<<endl;

                ProcessStart = clock();
                SegImgResult = PyObject_CallObject(RunFunc, ArgArray_Running);
                if(SegImgResult == nullptr){

                    cout<<"Testing Function Calling Error"<<endl;
                    PyErr_Print();
                    return -1;
                }
                ProcessEnd   = clock();
                cout<<"The Processing Time of Image "<< Index_i<< " Is :"<< (double)(ProcessEnd - ProcessStart)/ (double)CLOCKS_PER_SEC<< endl;

                Segmentations  = cv::Mat::zeros(RunningImage.rows, RunningImage.cols, CV_16UC1);
                FinalResult    = cv::Mat::zeros(RunningImage.rows, RunningImage.cols, CV_8UC1 );

                if(PyList_Check(SegImgResult)){

                    uint16_t NumListItems = PyList_Size(SegImgResult);

                    for(Index_j = 0; Index_j < NumListItems; Index_j++){

                        PyArrayObject *EachItem = (PyArrayObject *)PyList_GetItem(SegImgResult, Index_j);

                        int Rows  = EachItem->dimensions[0], Cols = EachItem -> dimensions[1];
                        FinalSegResult = cv::Mat::zeros(Rows, Cols, CV_8UC1);
                        ScaleImage     = cv::Mat::zeros(RunningImage.rows, RunningImage.cols, CV_8UC1 );

                        for(Index_m = 0; Index_m < Rows; Index_m++){

                            uint8_t *FinalSegPtr = FinalSegResult.ptr<uint8_t>(Index_m);
                            for(Index_n = 0; Index_n < Cols; Index_n++){

                                FinalSegPtr[Index_n] = *(uint8_t*)(EachItem->data + Index_m * EachItem->strides[0] + Index_n * EachItem->strides[1]);
                            }
                        }

                        sprintf(ImgFullSavingPath, "%s/Scale_%d/bin%.6d.png", SaveImgPath, Index_j, Index_i);
                        cv::imwrite(ImgFullSavingPath, FinalSegResult);

    //                    cv::imshow("ScaleResult", FinalSegResult);

                        cv::morphologyEx(FinalSegResult, FinalSegResult, cv::MORPH_CLOSE, cv::Mat());

    //                    cv::imshow("PostProcOne", FinalSegResult);

                        FinalSegResult.copyTo(HolesSearching);

                        cv::floodFill  (HolesSearching, cv::Point(0,0), UCHAR_MAX);

    //                    cv::imshow("PostProcTwo", HolesSearching);

                        cv::bitwise_not(HolesSearching, HolesSearching);

    //                    cv::imshow("PostProcThree", HolesSearching);

                        cv::bitwise_or (HolesSearching, FinalSegResult, FinalSegResult);

    //                    cv::imshow("PostProcFour", FinalSegResult);

                        cv::medianBlur (FinalSegResult, FinalSegResult, 9);

    //                    cv::imshow("PostProcFive", FinalSegResult);

                        cv::resize(FinalSegResult, ScaleImage, cv::Size(RunningImage.cols, RunningImage.rows), 0, 0, cv::INTER_NEAREST);

    //                    cv::imshow("Resize Image", ScaleImage);

    //                    cv::waitKey(0);
    //                    cv::destroyAllWindows();

                        for(Index_m = 0; Index_m < Segmentations.rows; Index_m++){

                            uint16_t *SegPtr = Segmentations.ptr<uint16_t>(Index_m);
                            uint8_t  *SclPtr = ScaleImage.ptr<uint8_t>(Index_m);
                            for(Index_n = 0; Index_n < Segmentations.cols; Index_n++){

                                SegPtr[Index_n] = SegPtr[Index_n] + SclPtr[Index_n];
                            }
                        }

                        FinalSegResult.release();
                        ScaleImage.release();
                        HolesSearching.release();
                    }

                    for(Index_m = 0; Index_m < FinalResult.rows; Index_m ++){

                        uint16_t *SegPtr = Segmentations .ptr<uint16_t>(Index_m);
                        uint8_t  *FRPtr  = FinalResult.ptr<uint8_t>(Index_m);
                        for(Index_n = 0; Index_n < FinalResult.cols; Index_n++){

                            if (SegPtr[Index_n] / 3  >= 85){

                                FRPtr[Index_n] = 255;
                            }
                        }
                    }

    //                cv::imshow("FinalResult", FinalResult);
    //                cv::waitKey(0);
    //                cv::destroyWindow("FinalResult");


                    FinalResult.copyTo(HolesSearching);
                    cv::floodFill  (HolesSearching, cv::Point(0,0), UCHAR_MAX);
                    cv::bitwise_not(HolesSearching, HolesSearching);
                    cv::bitwise_or (HolesSearching, FinalResult, FinalResult);
                    sprintf(ImgFullSavingPath, "%s/Result/bin%.6d.png", SaveImgPath, Index_i + 1);

                    cv::imwrite(ImgFullSavingPath, FinalResult);

                    sprintf(ImgFullSavingPath, "%s/Result/Before_bin%.6d.png", SaveImgPath, Index_i + 1);

                    cv::imwrite(ImgFullSavingPath, ROIImage);

                    Py_DECREF(SegImgResult);
                    Py_DECREF(ArgArray_Running);

                    HolesSearching.release();
                    FinalResult   .release();
                    Segmentations .release();
                    RunningImage  .release();
                    ROIImage      .release();
                    ScaleImage    .release();

                }else{

                    cout<<"Unexpected Return, It's Not a PyList Object"<<endl;
                    PyErr_Print();


                    Py_DECREF(SegImgResult);
                    Py_DECREF(ArgArray_Running);

                    RunningImage .release();
                    FinalResult  .release();
                    Segmentations.release();
                    ROIImage     .release();
                    ScaleImage   .release();

                    break;
                }
            }else{

                FinalResult    = cv::Mat::zeros(RunningImage.rows, RunningImage.cols, CV_8UC1 );
                sprintf(ImgFullSavingPath, "%s/Result/bin%.6d.png", SaveImgPath, Index_i);

                cv::imwrite(ImgFullSavingPath, FinalResult);

                RunningImage.release();
                FinalResult.release();
            }
        }

    }else{

        cout<<"This is not a List Object, Calling failed"<<endl;
    }

    Py_DECREF(GTImgPathList);
    Py_DECREF(RunFuncName);
    Py_DECREF(ImgPathList);


    Py_DECREF(pModule);
    Py_DECREF(pDict);
    Py_DECREF(pInstance);
    Py_Finalize();

    BkImage.release();


    return 0;
}

float AchieveTheParameter(char *ParamStr){

    uint32_t ParamLength = strlen(ParamStr), ParamResult = 0;
    float    FloatResult = 0;
    uint32_t Index_i = 0, Index_j = 0, Index_k = 0, ParamStep = 1, TotalStep = 1;

    bool IsFloatNum = false;
    for (Index_i = 2; Index_i < ParamLength; Index_i ++){

        ParamStep = 1;
        Index_j   = Index_i;

        while(Index_j < ParamLength && ParamStr[Index_j] >= '0' && ParamStr[Index_j] <= '9'){

            ParamStep *= 10;
            Index_j++;
        }

        ParamStep   /= 10;
        ParamResult += (ParamStep * ((int32_t)(ParamStr[Index_i] - '0')));

        if (ParamStep == 1){
            break;
        }
    }

    for (Index_i = Index_j; Index_i < ParamLength; Index_i ++){

        ParamStep = 1;
        Index_k   = Index_i;
        while(Index_k < ParamLength && ParamStr[Index_k] >= '0' && ParamStr[Index_k] <= '9'){

            ParamStep *= 10;
            Index_k ++;
        }

        if(TotalStep < ParamStep){

            TotalStep = ParamStep;
        }

        ParamStep /= 10;
        FloatResult += (ParamStep * ((int32_t)(ParamStr[Index_i] - '0')));

        if(ParamResult == 1){
            break;
        }
    }

    FloatResult /= TotalStep;
    FloatResult += ParamResult;
    return FloatResult;
}

void GenerateBackgroundImage(const vector<char *> ImgPath, const uint16_t NumOfBackImg, cv::Mat &BKImage){


    int16_t Index_i = 0, Index_j = 0, Index_k = 0, Index_m = 0, Index_n = 0;
    char ImgFilePath[512];
    cv::Mat TempImg;
    cv::Mat BkImg;
    uint8_t ***AllImagePixel;
    for(Index_k = 0; Index_k < NumOfBackImg; Index_k++){

        TempImg = cv::imread(ImgPath.at(Index_k), CV_LOAD_IMAGE_COLOR);
        if(Index_k == 0){

            AllImagePixel = new uint8_t **[TempImg.rows];
            for(Index_i = 0; Index_i < TempImg.rows; Index_i++){
                AllImagePixel[Index_i] = new uint8_t *[TempImg.cols];

                for(Index_j = 0; Index_j < TempImg.cols; Index_j++){
                    AllImagePixel[Index_i][Index_j] = new uint8_t[NumOfBackImg];
                }
            }

            BkImg = cv::Mat(TempImg.rows, TempImg.cols, CV_8UC1);
        }

        for(Index_i = 0; Index_i < TempImg.rows; Index_i++){

            uint8_t *TImg_ptr = TempImg.ptr<uint8_t>(Index_i);

            for(Index_j = 0; Index_j < TempImg.cols; Index_j++){

                AllImagePixel[Index_i][Index_j][Index_k] = uint8_t(0.299 * TImg_ptr[3 * Index_j] + 0.587 * TImg_ptr[3 * Index_j + 1] + 0.114 * TImg_ptr[3 * Index_j + 2]);
            }
        }
        TempImg.release();
    }

    uint8_t temp = 0;
    for(Index_i = 0; Index_i < BkImg.rows; Index_i++){

        for(Index_j = 0; Index_j < BkImg.cols; Index_j++){

            for(Index_m = 1; Index_m < (NumOfBackImg); Index_m++){

                for(Index_n = Index_m; Index_n > 0  && AllImagePixel[Index_i][Index_j][Index_n] < AllImagePixel[Index_i][Index_j][Index_n - 1]; Index_n--){

                    temp = AllImagePixel[Index_i][Index_j][Index_n];
                    AllImagePixel[Index_i][Index_j][Index_n] = AllImagePixel[Index_i][Index_j][Index_n - 1];
                    AllImagePixel[Index_i][Index_j][Index_n - 1] = temp;
                }
            }
        }
    }

    for(Index_i = 0; Index_i < BkImg.rows; Index_i++){

        uint8_t *BkPtr = BkImg.ptr<uint8_t>(Index_i);
        for(Index_j = 0; Index_j < BkImg.cols; Index_j++){

            BkPtr[Index_j] = AllImagePixel[Index_i][Index_j][uint16_t((NumOfBackImg) / 2)];
        }
    }


    BkImg.copyTo(BKImage);

    for(Index_i = 0; Index_i < BkImg.rows; Index_i++){

        for(Index_j = 0; Index_j < BkImg.cols; Index_j++){

            delete [] AllImagePixel[Index_i][Index_j];
        }
    }

    for(Index_i = 0; Index_i < BkImg.rows; Index_i++){
        delete []AllImagePixel[Index_i];
    }

    delete [] AllImagePixel;

    BkImg.release();

}
