//RESLA-AUTONOMOUS ROBOT PROJECT FOR RTES COURSE, CU BOULDER
//AUTHORS: PRAVEEN GNANASEKARAN, RISHABH BERLIA
//THE CODE CONTAINS 5 THREADS NAMELY A SCHEDULER WHICH SPAWNS 4 OTHER THREADS WHICH ARE USED IN THE CRITICAL FUNCTIONING OF THE SERVICES
// HEADER FILES TO SUPPORT THE OPEN CV LIBRARIES, MACHINE LEARNING AND THE STRING CONVERTERS TO READ/ WRITE TO FILES AND IMAGES

#include <pthread.h>
#include <semaphore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/detection_based_tracker.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <fcntl.h>
#include <termios.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
using namespace cv;
using namespace std;

using namespace cv::ml;
//MACROS USED IN THE PROGRAM
#define NUM_THREADS (5)
//SETTING AFFINITY TO ONE CORE
#define NUM_CPUS (1)
#define NSEC_PER_SEC (1000000000)
#define NSEC_PER_MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)

//SET SIZES FOR EACH IMAGE
int SET1_SIZE=13;    //LEFT
int SET2_SIZE=10;    //STOP IS 22 , NO ENTRY IS 36
int SET3_SIZE=10;    //RIGHT
int SET4_SIZE=13;    //NEGATIVE


#define ROW_SIZE (SET2_SIZE+SET3_SIZE+SET4_SIZE)*30

sem_t sem_feature,sem_classify,sem_motor,sem_ultrasonic;
//DIFFERENT CLASSES FOR CLASSIFYING THE IMAGE SETS
int LEFT_CLASS=0;
int STOP_CLASS=1;
int RIGHT_CLASS=2;
int NEGATIVE_CLASS=3;

//SVM INITIALIZATION
static Ptr<SVM> svm = SVM::create();
vector<float> featureVector;
long long int cnttt=0;
int total_count;
double aver_freq=1;
//REQUEST FREQUENCY VARIABLES
struct timespec feat_thread_request1 = {0, 0};
struct timespec feat_thread_request2 = {0, 0};
struct timespec feat_threaddt = {0, 0};
double feat_avg;

struct timespec class_thread_request1 = {0, 0};
struct timespec class_thread_request2 = {0, 0};
struct timespec class_threaddt = {0, 0};
double class_avg;

struct timespec ultra_thread_request1 = {0, 0};
struct timespec ultra_thread_request2 = {0, 0};
struct timespec ultra_threaddt = {0, 0};
double ultra_avg;

struct timespec motion_thread_request1 = {0, 0};
struct timespec motion_thread_request2 = {0, 0};
struct timespec motion_threaddt = {0, 0};
double motion_avg;

//STRUCT FOR THE CREATION OF PTHREADS USED IN THE CODE
typedef struct a
{
    int threadIdx;
} threadParams_t;

threadParams_t threadParams[NUM_THREADS];

// POSIX thread declarations and scheduling attributes
pthread_t threads[NUM_THREADS], schedularThread;
//threadParams_t threadParams[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
pthread_attr_t main_attr;
int rt_max_prio, rt_min_prio;
int rt_max_prio1, rt_min_prio1;
struct sched_param rt_param[NUM_THREADS];
struct sched_param main_param,main_param1;
struct sched_param thrd1,thrd2;
pthread_mutex_t feature_mutexlock,image_mutexlock;
pthread_mutexattr_t mutexattr;
pid_t mainpid,mainpid1;
Mat image;
CvCapture* capture;

//---- SERIAL COMMUNICATION WITH THE ARDUINO
int fd, n, i;
struct termios toptions;
int stop_flag,turn_left_flag,turn_right_flag;
int class_arr[3];

//delta_t() IS BORROWED FROM SAM SIEWERT'S EXAMPLE CODES AND IS USED TO CALCULATE THE DIFFERNECE BETWEEN THE STOP AND START TIMES.
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }

  return(1);
}
//This function is borrowed from Sam Siewert's exmaples from Exercise 1
void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
     case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
     case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n");
       break;
     case SCHED_RR:
           printf("Pthread Policy is SCHED_OTHER\n");
           break;
     default:
       printf("Pthread Policy is UNKNOWN\n");
   }

}
//THREAD THAT READS THE ULTRASONIC DATA FROM ARDUINO AND MAKES AN INFORMED DECISION ON THE VALUE READ
void *read_ultrasonic(void *threadp)
{
while(1){
    //read ultrasonic data and set stop_flag
    sem_wait(&sem_ultrasonic);
    int policy;

        struct timespec start_time = {0, 0};
        struct timespec finish_time = {0, 0};
        struct timespec thread_dt={0,0};

        clock_gettime(CLOCK_REALTIME, &start_time);
        printf("\nThread ULTRA REQUEST TIME %ld sec, %lf msec (%ld microsec)\n", start_time.tv_sec, (double)((double)start_time.tv_nsec / NSEC_PER_MSEC), (start_time.tv_nsec / NSEC_PER_MICROSEC));
        clock_gettime(CLOCK_REALTIME, &ultra_thread_request2);
        delta_t(&ultra_thread_request2, &ultra_thread_request1, &ultra_threaddt);
        clock_gettime(CLOCK_REALTIME, &ultra_thread_request1);

        threadParams_t *threadParams = (threadParams_t *)threadp;
        struct sched_param param;

        char buf[4];
         n= write(fd, "U", 1);
        if(n<0)printf("FAILED\n\r");
        read(fd,buf,4);

//    with buf value convert to integer and determine if stop_flag should be set
    int distance=atoi(buf);
    printf("%d ", distance);
    if(distance> 300)
    stop_flag=1;	//OBJECT IS TOO CLOSE, SO STOP
    pthread_getschedparam(threads[4],&policy,&param);
//TIMING TRACING FOR THE THREAD
    clock_gettime(CLOCK_REALTIME, &finish_time);
    printf("\nThread ULTRA COMPLETION TIME %ld sec, %lf msec (%ld microsec)\n", finish_time.tv_sec, (double)((double)finish_time.tv_nsec / NSEC_PER_MSEC), (finish_time.tv_nsec / NSEC_PER_MICROSEC));
delta_t(&finish_time, &start_time, &thread_dt);    //compute the time of thread execution from the start and end times
   printf("\nThread ULTRA idx=%d (Priority: %d) ran %ld sec, %lf msec (%ld microsec)\n", threadParams->threadIdx,param.sched_priority, thread_dt.tv_sec, (double)((double)thread_dt.tv_nsec / NSEC_PER_MICROSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));
   ultra_avg=(ultra_avg+(1/((double)((double)ultra_threaddt.tv_nsec / NSEC_PER_MSEC))*1000))/aver_freq;

   //REQUEST FREQUENCY IS 1/TIME_PERIOD
   printf("ULTRA average request frequency is %lf Hz\n",(1/((double)((double)ultra_threaddt.tv_nsec / NSEC_PER_MSEC))*1000));

//SYNCHRONIZE ACCESS TO NEXT THREAD
        sem_post(&sem_motor);
    }

}
//THREAD WITH LEAST PRIORITY USED TO SEND MOTOR CONTROL TO THE BOT AFTER ALL OTHER CLASSIFICATIONS HAVE BEEN DONE
void *motor_control(void *threadp)
{
    while(1){
    //SEMAPHORES USED TO SYNCHRONIZE THE THREAD SO THAT THE THREAD IS ONLY ACCESSED AFTER AN INFORMED DECISION HAS BEEN MADE
    sem_wait(&sem_motor);
    total_count++;
    int policy;

        struct timespec start_time = {0, 0};
        struct timespec finish_time = {0, 0};
        struct timespec thread_dt={0,0};

        clock_gettime(CLOCK_REALTIME, &start_time);
        printf("\nThread MOTOR REQUEST TIME %ld sec, %lf msec (%ld microsec)\n", start_time.tv_sec, (double)((double)start_time.tv_nsec / NSEC_PER_MSEC), (start_time.tv_nsec / NSEC_PER_MICROSEC));
        clock_gettime(CLOCK_REALTIME, &motion_thread_request2);
        delta_t(&motion_thread_request2, &motion_thread_request1, &motion_threaddt);
        clock_gettime(CLOCK_REALTIME, &motion_thread_request1);

        threadParams_t *threadParams = (threadParams_t *)threadp;
        struct sched_param param;

        char buf[2];
//    if(total_count%6==0)
//    {
    	int max;
        //find max of class_arr[]
        if(class_arr[0]>=class_arr[1]){
            if(class_arr[0]>=class_arr[2])
            max=0;
            else if(class_arr[2]>class_arr[0])
            max=2;
            }
        else if(class_arr[1]>class_arr[0])
        {
            if(class_arr[1]>=class_arr[2])
            max=1;
            else if(class_arr[2]>class_arr[1])
            max=2;
        }
        class_arr[0]=class_arr[1]=class_arr[2]=0;
        if(max==2)
        {

        n= write(fd, "F", 1);    //send_forward command

        }
        else if(max==0)
        {

            n= write(fd, "S", 1);    //send_forward command

            stop_flag=0;
        }

        else if(max==1)
        {

            n= write(fd, "R", 1);    //send_forward command

            turn_right_flag=0;
        }

        if(n<0)printf("SEND FAILED\n\r");
        read(fd,buf,2);

    total_count=0;
//    }

//TIME TRACING FOR THE THREAD
    pthread_getschedparam(threads[3],&policy,&param);
    clock_gettime(CLOCK_REALTIME, &finish_time);
    printf("\nThread MOTOR COMPLETION TIME %ld sec, %lf msec (%ld microsec)\n", finish_time.tv_sec, (double)((double)finish_time.tv_nsec / NSEC_PER_MSEC), (finish_time.tv_nsec / NSEC_PER_MICROSEC));
    delta_t(&finish_time, &start_time, &thread_dt);    //compute the time of thread execution from the start and end times
   printf("\nThread MOTOR idx=%d (Priority: %d) ran %ld sec, %lf msec (%ld microsec)\n", threadParams->threadIdx,param.sched_priority, thread_dt.tv_sec, (double)((double)thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));
   motion_avg=(ultra_avg+(1/((double)((double)motion_threaddt.tv_nsec / NSEC_PER_MSEC))*1000))/aver_freq;

   //REQUEST FREQUENCY IS 1/TIME_PERIOD
   printf("MOTION average request frequency is %lf Hz\n",(1/((double)((double)motion_threaddt.tv_nsec / NSEC_PER_MSEC))*1000));

   aver_freq++;
	sem_post(&sem_feature);
    }
}
//THREAD WHICH CLASSIFIES IMAGE BASED ON TRAINED SVM
void *image_classification(void *threadp)
{
struct timespec start_time = {0, 0};
struct timespec finish_time = {0, 0};
struct timespec thread_dt={0,0};
    while(1){
	//USED TO SYNCHRONIZE ACCESSS BETWEEN THE THREADS
    sem_wait(&sem_classify);
    clock_gettime(CLOCK_REALTIME, &start_time);
    printf("\nThread CLASS REQUEST TIME %ld sec, %lf msec (%ld microsec)\n", start_time.tv_sec, (double)((double)start_time.tv_nsec / NSEC_PER_MSEC), (start_time.tv_nsec / NSEC_PER_MICROSEC));
    clock_gettime(CLOCK_REALTIME, &class_thread_request2);
    delta_t(&class_thread_request2, &class_thread_request1, &class_threaddt);
    clock_gettime(CLOCK_REALTIME, &class_thread_request1);

    threadParams_t *threadParams = (threadParams_t *)threadp;
	struct sched_param param;
    float result;
    int policy;
    //mutex locks are used to protect shared resoucres between this thread and the feature extraction thread,
    pthread_mutex_lock(&feature_mutexlock);
    result=svm->predict(featureVector);
    pthread_mutex_unlock(&feature_mutexlock);
    pthread_getschedparam(threads[2],&policy,&param);
    //PRINTING THE OUTPUT OF THE PREDICTED IMAGE FOR THE USER TO SEE.
    if(result==STOP_CLASS){
        stop_flag=1;
        class_arr[0]=class_arr[0]+1;
    cout<<"STOP SIGN DETECTED"<<endl;
        }
    else if(result==RIGHT_CLASS){
        cout<<"RIGHT SIGN DETECTED"<<endl;
                class_arr[1]=class_arr[1]+1;
        turn_right_flag=1;
        }
    else if(result==LEFT_CLASS)
        {
        cout<<"LEFT SIGN DETECTED"<<endl;
        turn_left_flag=1;
        }
    else{
        cout<<"NO SIGN DETECTED"<<endl;
            class_arr[2]=class_arr[2]+1;
}
    cvWaitKey(33);
    //SHOW THE IMAGE SO IT IS VISIBLE ON THE SCREEN
    pthread_mutex_lock(&image_mutexlock);
    namedWindow( "Feature window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Feature window", image );                   // Show our image inside it.

    image.release();
    pthread_mutex_unlock(&image_mutexlock);
    clock_gettime(CLOCK_REALTIME, &finish_time);
    printf("\nThread CLASS COMPLETION TIME %ld sec, %lf msec (%ld microsec)\n", finish_time.tv_sec, (double)((double)finish_time.tv_nsec / NSEC_PER_MSEC), (finish_time.tv_nsec / NSEC_PER_MICROSEC));
    // TIMING TRACES
    delta_t(&finish_time, &start_time, &thread_dt);    //compute the time of thread execution from the start and end times
    printf("\nThread CLASS idx=%d (Priority: %d) ran %ld sec, %lf msec (%ld microsec)\n", threadParams->threadIdx,param.sched_priority, thread_dt.tv_sec, (double)((double)thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));

    class_avg=(class_avg+(1/((double)((double)class_threaddt.tv_nsec / NSEC_PER_MSEC))*1000))/aver_freq;

    //REQUEST FREQUENCY IS 1/TIME_PERIOD
    printf("CLASS average request frequency is %lf Hz\n",(1/((double)((double)class_threaddt.tv_nsec / NSEC_PER_MSEC))*1000));

    sem_post(&sem_ultrasonic);


    }
}
// THREAD TO EXTRACT FEATURES FROM THE INPUT IMAGES
void *feature_extraction(void *threadp)
{
struct timespec start_time = {0, 0};
struct timespec finish_time = {0, 0};
struct timespec thread_dt={0,0};


    while(1){
    	//SEMAPHORES ARE USED TO SYNCHRONIZE ACCESS BETWEEN THREADS
            sem_wait(&sem_feature);

            clock_gettime(CLOCK_REALTIME, &start_time);
            printf("\nThread FEAT REQUEST TIME %ld sec, %lf msec (%ld microsec)\n", start_time.tv_sec, (double)((double)start_time.tv_nsec / NSEC_PER_MSEC), (start_time.tv_nsec / NSEC_PER_MICROSEC));
            clock_gettime(CLOCK_REALTIME, &feat_thread_request2);
            delta_t(&feat_thread_request2, &feat_thread_request1, &feat_threaddt);
            clock_gettime(CLOCK_REALTIME, &feat_thread_request1);
            threadParams_t *threadParams = (threadParams_t *)threadp;
                struct sched_param param;

            IplImage* frame;
            //THE FEATURE EXTRACTION WAS DONE USING HISTOGRAM OF ORIENTED GRADIENTS(HOG)
            HOGDescriptor hog;//(Size(40,40),Size(10,10),Size(5,5),Size(5,5),8,1,-1,0,0.2,false,64,true);
            int policy;
            vector<Point> locations;
            //DIFFERENT PARAMETERS FOR HOG WHICH WORKED BEST FOR CLASSIFICATION
            hog.winSize = Size(40, 40);
            hog.cellSize=Size(5,5);
            hog.blockSize=Size(10,10);

            hog.blockStride=Size(5,5);
            hog.nbins=8;
    //        hog.signedGradient =false;
            hog.winSigma= 4;
            hog.histogramNormType=0;
            hog.L2HysThreshold=2.0000000000000001e-01;

            frame=cvQueryFrame(capture);

             if(!frame) cout<<"FRAME ERROR";

             pthread_mutex_lock(&image_mutexlock);
            image=(cvarrToMat(frame));

            namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
            imshow( "Display window", image );                   // Show our image inside it.
            // RESIZE IMAGE TO MAINTAIN UNIFORMITY
            resize(image,image,Size(40,40),0,0,INTER_LINEAR);
            pthread_mutex_unlock(&image_mutexlock);

                if(! image.data )                              // Check for invalid input
                {
                    cout <<  "Could not open or find the image" << std::endl ;
                }
			// MUTEX LOCK TO SHARE BETWEEN THIS THREAD AND IMAGE CLASSIFY THREAD
            pthread_mutex_lock(&feature_mutexlock);
            //EXTRACT HOG FEATURES
            hog.compute(image,featureVector,Size(5,5),Size(1,1),locations);
            pthread_mutex_unlock(&feature_mutexlock);
            clock_gettime(CLOCK_REALTIME, &finish_time);
            printf("\nThread FEAT COMPLETION TIME %ld sec, %lf msec (%ld microsec)\n", finish_time.tv_sec, (double)((double)finish_time.tv_nsec / NSEC_PER_MSEC), (finish_time.tv_nsec / NSEC_PER_MICROSEC));
            delta_t(&finish_time, &start_time, &thread_dt);    //compute the time of thread execution from the start and end times
            pthread_getschedparam(threads[1],&policy,&param);

            printf("\nThread FEAT idx=%d (Priority: %d) ran %ld sec, %lf msec (%ld microsec)\n", threadParams->threadIdx,param.sched_priority, thread_dt.tv_sec, (double)((double)thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));

            feat_avg=(feat_avg+(1/((double)((double)feat_threaddt.tv_nsec / NSEC_PER_MSEC))*1000))/aver_freq;

            //REQUEST FREQUENCY IS 1/TIME_PERIOD
            printf("FEAT average request frequency is %lf Hz Request time is \n",(1/((double)((double)feat_threaddt.tv_nsec / NSEC_PER_MSEC))*1000),((double)((double)feat_threaddt.tv_nsec / NSEC_PER_MSEC))*1000);
            sem_post(&sem_classify);
    }

}
// THIS IS THE CALLBACK FUNCTION OF THE SCHEDULER THREAD WHICH CREATES THE THREE OTHER THREADS. IT HAS A PRIORITY OF 99
void *scheduler(void *threadp)
{

    printf("\n-----------------------Start Scheduler-------------------------------\n");
    int policy;
    struct sched_param param;
    int sum=0, i,rc;
    //Local Variables
    pthread_t thread;
    cpu_set_t cpuset;

    struct timespec start_time = {0, 0};
	struct timespec finish_time = {0, 0};
	struct timespec thread_dt={0,0};

	clock_gettime(CLOCK_REALTIME, &feat_thread_request1);
	clock_gettime(CLOCK_REALTIME, &class_thread_request1);
	clock_gettime(CLOCK_REALTIME, &ultra_thread_request1);
	clock_gettime(CLOCK_REALTIME, &motion_thread_request1);

    threadParams_t *threadParams = (threadParams_t *)threadp;

    CPU_ZERO(&cpuset);                    // setting the CPU cores to 0

    capture = (CvCapture *)cvCreateCameraCapture(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 480);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 320);

    sem_post(&sem_feature);
//ORDER OF PRIORITY- FEATURE EXTRACTION, IMAGE CLASSIFICATION, DISTANCE MEASUREMENT AND MOTOR CONTROL
    pthread_create(&threads[1],   // pointer to thread descriptor
                      (const pthread_attr_t *)&rt_sched_attr[1],     // use default attributes
                      feature_extraction, // thread function entry point
                      (void *)&(threadParams[1]) // parameters to pass in
                     );
    pthread_create(&threads[2],   // pointer to thread descriptor
                      (const pthread_attr_t *)&rt_sched_attr[2],     // use default attributes
                     image_classification, // thread function entry point
                      (void *)&(threadParams[2]) // parameters to pass in
                     );

    pthread_create(&threads[3],   // pointer to thread descriptor
                      (const pthread_attr_t *)&rt_sched_attr[3],     // use default attributes
                     read_ultrasonic, // thread function entry point
                      (void *)&(threadParams[3]) // parameters to pass in
                     );
    pthread_create(&threads[4],   // pointer to thread descriptor
                      (const pthread_attr_t *)&rt_sched_attr[4],     // use default attributes
                       motor_control, // thread function entry point
                      (void *)&(threadParams[4]) // parameters to pass in
                     );

    clock_gettime(CLOCK_REALTIME, &start_time);        //obtatining the start time before the synthetic load generation.

    printf("\nThread idx=%d (Priority: %d) ran %ld sec, %lf msec (%ld microsec)\n", threadParams->threadIdx,param.sched_priority, thread_dt.tv_sec, (double)((double)thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));

       printf("\n-----------------------End Scheduler-------------------------------\n");

//    }

    pthread_join(threads[1],NULL);                //join the pthreads fib10 and fib20.
    pthread_join(threads[2],NULL);
    pthread_join(threads[3],NULL);
    pthread_join(threads[4], NULL);
    clock_gettime(CLOCK_REALTIME, &finish_time);

    pthread_getschedparam(threads[0],&policy,&param);        //get the scheduling parameters from the calling thread
    delta_t(&finish_time, &start_time, &thread_dt);        //compute the time of thread execution from the start and end times
    printf("\nThread idx=%d (Priority: %d) ran %ld sec, %lf msec (%ld microsec)\n", threadParams->threadIdx,param.sched_priority, thread_dt.tv_sec, (double)((double)thread_dt.tv_nsec / NSEC_PER_MSEC), (thread_dt.tv_nsec / NSEC_PER_MICROSEC));

}
//MAIN FUNCTION
int main( int argc, char** argv )
{
    int rc;
    int i, scope;
    cpu_set_t cpuset;
    //IMAGENAME TO READ THE EXTRACTED HOG FEATURES FROM DISK
    string imageName("/home/praveen/Desktop/Custom/");
    CPU_ZERO(&cpuset);            //setting the CPU cores of all cores to 0.
    for(i=0; i < NUM_CPUS; i++)
    CPU_SET(i, &cpuset);

    mainpid=getpid();            //get the thread id of the calling thread.
    //INITILIAZE SEMAPHORES TO SYNCHRONIZE ACCESS BETWEEN THREADS
    sem_init(&sem_feature, 0, 0);
    sem_init(&sem_classify, 0, 0);
    sem_init(&sem_motor, 0, 0);
    sem_init(&sem_ultrasonic,0,0);
//------------------------

    /* open serial port TO COMMUNICATE WITH THE ARDUINO UNO TO SEND AND RECEIVE DATA */
    fd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY);
    printf("fd opened as %i\n", fd);
    /* get current serial port settings */
    tcgetattr(fd, &toptions);
    /* set 9600 baud both ways */
    cfsetispeed(&toptions, B9600);
    cfsetospeed(&toptions, B9600);
    /* 8 bits, no parity, no stop bits */
    //COMMON TERMIOS FLAGS TO COMMUNICATE TO ANY DEVICE
    toptions.c_cflag &= ~PARENB;
    toptions.c_cflag &= ~CSTOPB;
    toptions.c_cflag &= ~CSIZE;
    toptions.c_cflag |= CS8;
    /* Canonical mode */
    toptions.c_lflag |= ICANON;
    /* commit the serial port settings */
    tcsetattr(fd, TCSANOW, &toptions);
    fcntl(fd,F_SETFL, FNDELAY);    // read function made non blocking
    //---- SERIAL COMM

//---------------

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);        //max priority of the SCHED_FIFO
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);        //min priority of the SCHED_FIFO

    printf("\nBefore Adjustments to Schedule Policy:");
    print_scheduler(); //print scheduler before assigning SCHED_FIFO
    rc=sched_getparam(mainpid, &main_param);            //get the scheduling parameters of the thread and transferring it to main_param
    main_param.sched_priority=rt_max_prio;            //setting the max priority of the calling thread to 99.
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);    //set the SCHED_FIFO scheduler of the main_param.
    if(rc < 0) perror("main_param");
    printf("\nAfter Adjustments to Schedule Policy:");
    print_scheduler();    //print scheduler after assigning SCHED_FIFO

    pthread_attr_getscope(&main_attr, &scope);//obtain the scope of the main_attr and print it
    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    //Attribute settings for the Threads
       for(i=0; i <NUM_THREADS; i++)
       {
           rc=pthread_attr_init(&rt_sched_attr[i]);            // intializing the pthread attributes for the three threads.
           rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);// set to explicit schedule policy and later to SCHED_FIFO
           rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);    //set the schedule policy of the three threads to SCHED_FIFO
           rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &cpuset);//set the affinity of the CPU cores to zero

           rt_param[i].sched_priority=rt_max_prio-i;            //set the priorities of the three threads as 99,98 and 97 respectively.
           pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);    //set the scheduling parameters of the three pthreads.
           threadParams[i].threadIdx=i;                    //fill the threadParams with their corresponding numbers

       }

       printf("rt_max_prio=%d\n", rt_max_prio);
       printf("rt_min_prio=%d\n", rt_min_prio);
       //FEATURE VECTOR FOR THE DATASET
       float **feature_arr= new float*[ROW_SIZE];
           for(int i=0;i<ROW_SIZE;i++)
               feature_arr[i]=new float[1568];
           int classes_arr[ROW_SIZE];
           cout<<ROW_SIZE<<" DATA SETS"<<endl;
               vector<double> rows;
               Mat classes;
               float value;
               string str_inter,str_outer ;
               ostringstream inter_convert,outer_convert;   // stream used for the conversion
// LOADING DATASETS INTO FEATURE VECTORS FOR ALL THE IMAGES
           //SET1 --> LEFT, NUMBER 34 FOR THE LEFT SIGN
           for(int outer=0;outer<SET1_SIZE;outer++){
               for(int inter=0;inter<=29;inter++){

                   inter_convert << inter;
                   str_inter = inter_convert.str();
                   outer_convert << outer;
                   str_outer = outer_convert.str();

                   string path = imageName+"00034/000"+((outer<=9)?("0"+str_outer):(str_outer))+"_000"+((inter<=9)?("0"+str_inter):(str_inter))+".txt";

                   ifstream myfile(path.c_str());
                   inter_convert.str(std::string());
                   outer_convert.str(std::string());

                   if(!myfile)
                       cout<<"ERROR1 ";
                   else{

                       for(int cnt=0;cnt<1569;cnt++){
                           myfile>>value;
                           feature_arr[cnttt][cnt]=value;
                           }
                       classes_arr[cnttt]=LEFT_CLASS;    //0 is LEFT
                       cnttt++;

                       }
               }
           }
           //SET2 --> STOP, NUMBER 14 FPR STOP SIGN
           for(int outer=0;outer<SET2_SIZE;outer++){
               for(int inter=0;inter<=29;inter++){

                   inter_convert << inter;      // insert the textual representation of 'Number' in the characters in the stream
                   str_inter = inter_convert.str(); // set 'Result' to the contents of the stream
                   outer_convert << outer;
                   str_outer = outer_convert.str();

                   string path1 = imageName+"STOP/000"+((outer<=9)?("0"+str_outer):(str_outer))+"_000"+((inter<=9)?("0"+str_inter):(str_inter))+".txt";

                   ifstream myfile1(path1.c_str());
                   inter_convert.str(std::string());
                   outer_convert.str(std::string());
                   if(!myfile1)
                       cout<<"ERROR2 FOR OUTER  "<<outer;
                   else{
                       for(int cnt=0;cnt<1569;cnt++){
                           myfile1>>value;
                           feature_arr[cnttt][cnt]=value;
                           }

                       classes_arr[cnttt]=STOP_CLASS;    //1 is STOP sign
                       cnttt++;
                   }
               }
           }
           //SET3 --> RIGHT, NUMBER 33 FPR RIGHT SIGN
           for(int outer=0;outer<SET3_SIZE;outer++){
               for(int inter=0;inter<=29;inter++){

                   inter_convert << inter;      // insert the textual representation of 'Number' in the characters in the stream
                   str_inter = inter_convert.str(); // set 'Result' to the contents of the stream
                   outer_convert << outer;
                   str_outer = outer_convert.str();

                   string path1 = imageName+"RIGHT/000"+((outer<=9)?("0"+str_outer):(str_outer))+"_000"+((inter<=9)?("0"+str_inter):(str_inter))+".txt";

                   ifstream myfile1(path1.c_str());
                   inter_convert.str(std::string());
                   outer_convert.str(std::string());
                   if(!myfile1)
                       cout<<"ERROR3 FOR OUTER "<<outer;
                   else{
                       for(int cnt=0;cnt<1569;cnt++){
                           myfile1>>value;
                           feature_arr[cnttt][cnt]=value;
                           }

                       classes_arr[cnttt]=RIGHT_CLASS;    //2 is RIGHT sign
                       cnttt++;
                   }
               }
           }
				   //SET4 --> NEAGTIVES, NUMBER 19 FOR NEGATIVE IMAGES
				   for(int outer=0;outer<SET4_SIZE;outer++){
					   for(int inter=0;inter<=29;inter++){

						   inter_convert << inter;      // insert the textual representation of 'Number' in the characters in the stream
						   str_inter = inter_convert.str(); // set 'Result' to the contents of the stream
						   outer_convert << outer;
						   str_outer = outer_convert.str();

						   string path1 = imageName+"NEGATIVE/000"+((outer<=9)?("0"+str_outer):(str_outer))+"_000"+((inter<=9)?("0"+str_inter):(str_inter))+".txt";

						   ifstream myfile1(path1.c_str());
						   inter_convert.str(std::string());
						   outer_convert.str(std::string());
						   if(!myfile1)
							   cout<<"ERROR4 FOR OUTER "<<outer;
						   else{
							   for(int cnt=0;cnt<1569;cnt++){
								   myfile1>>value;
								   feature_arr[cnttt][cnt]=value;
								   }

							   classes_arr[cnttt]=NEGATIVE_CLASS;    //2 is RIGHT sign
							   cnttt++;
						   }
					   }
				   }
			Mat featuresMat(ROW_SIZE,1568,CV_32F,feature_arr);
		   Mat classesMat(ROW_SIZE,1,CV_32SC1,&classes_arr);

		   //MODIFY SVM KERNEL VALUES TO SEE WHICH ONE BEST SUITS YOUR NEEDS
                svm->setType(SVM::C_SVC);
                svm->setC(50);    //50 with INTER 20000 best so far

                svm->setKernel(SVM::INTER);//RBF with 35 works well; INTER 200,20000
                svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 2000, 1e-6));
                cout<<"TRAINING.."<<endl;
                svm->train(featuresMat, ROW_SAMPLE, classesMat);
                cout<<"DONE TRAINING.."<<endl;

                // CREATE PTHREAD AND JOIN SO THE PRIORITY PREEMPTIVE SERVICE CAN RUN IN REAL TIME
       // create the pthread of the scheduler which will inturn create the two lower priority threads, fib10 and fib20.
       pthread_create(&threads[0],   // pointer to thread descriptor
                         (const pthread_attr_t *)&rt_sched_attr[0],     // use default attributes
                         scheduler, // thread function entry point
                         (void *)&(threadParams[0]) // parameters to pass in
                        );

       pthread_join(threads[0], NULL);    //join the pthread wiith the existing processes

    return 0;
}
