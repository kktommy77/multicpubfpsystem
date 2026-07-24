#include "cJSON/cJSON.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <stdint.h>
#include <float.h>
//#include "example_bsfp.c"

//#include <map>
//#include <iostream>

#define max(x, y) (x) > (y) ? (x) : (y)
#define min(x, y) (x) > (y) ? (y) : (x)

struct BFP {
    int g;
    int  max_exp;
    int mant_array[16];// mantissa
    bool sign_array[16]; 

};

struct bfp_data{
    bool sign;
    int exp;
    int mant;
};

struct BFPImage{

    int B;
    int C;
    int H;
    int W;
    struct bfp_data *data;
};



struct Image {
    int B;
    int H;
    int W;
    int C;
    float* data;
};

struct Image * Conv(struct Image * IN, int Cin, int Cout, int kernel_size, int stride, int padding,  cJSON* cjsonp, const char* name)
{
    int B= IN->B;
    int C= IN->C;
    int H= IN->H;
    int W= IN->W;

    struct Image* OUT = (struct Image* )malloc(sizeof(struct Image));
    OUT->B = B;
    OUT->C = Cout;
    OUT->H = (H-kernel_size+2*padding)/stride +1;
    OUT->W = (W-kernel_size+2*padding)/stride +1;
    int Hout = OUT->H;
    int Wout = OUT->W;

    OUT->data = (float*)malloc(sizeof(float)*B*Cout*Hout*Wout);


    char * weight_name= (char*)malloc(sizeof(char)*(50));
    char * bias_name = (char*)malloc(sizeof(char)*(50));

    int weight_size = Cout*Cin*kernel_size*kernel_size;
    int bias_size= Cout;
    int bias_exist = 0;
    float * weight = (float*)malloc(sizeof(float)*weight_size);
    float * bias =  (float*)malloc(sizeof(float)*bias_size);

    //////WEIGHT INITIALIZATION
    int b= kernel_size*kernel_size;
    int a= b*Cin;
    int c= kernel_size;
    
    strcpy(weight_name, name);
    strcat(weight_name, ".weight");
    cJSON * layer_weight = cJSON_GetObjectItem(cjsonp, weight_name);
    if((cJSON_GetArraySize(layer_weight)!=Cout)||(cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0))!=Cin)||(cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0))!=kernel_size)||(cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0), 0))!=kernel_size))
    {
        printf("Current Weight Matrix size is  %d x %d x %d x %d , But Real Weight Matrix size is %d x %d x %d x %d !\n", cJSON_GetArraySize(layer_weight), cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0)),cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0)), cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0), 0)), Cout, Cin, kernel_size, kernel_size);
        printf("Size does not match!\n");
        return NULL;
    }
    // i = a*Cin*kernel_size*kernel_size + b*kernel_size*kernel_size + 
    #pragma omp parallel for
    for(int i=0; i<weight_size; i++)
    {
        cJSON* value = cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, i/a), (i/b)%Cin), (i/c)%c), i%c);
        weight[i] = (float)(value->valuedouble);
    }
   // printf("WEight %f\n", weight[1]);
    //////BIAS INITIALIZATION
    strcpy(bias_name, name);
    strcat(bias_name, ".bias");
    cJSON * layer_bias = cJSON_GetObjectItem(cjsonp, bias_name);
    if(layer_bias != NULL)
    {
        bias_exist =1;
        if(cJSON_GetArraySize(layer_bias)!=Cout)
        {
            printf("Current Bias size is  %d , But Real Bias size is %d!\n", cJSON_GetArraySize(layer_bias), Cout);
            printf("Size does not match!\n");
            return NULL;
        }
        #pragma omp parallel for
        for(int i=0; i<bias_size; i++)
        {
            cJSON* value = cJSON_GetArrayItem(layer_bias, i);
            bias[i] = (float)(value->valuedouble);
        }

    }
    // PADDING OPERATION
    int PADDED_H = H+2*padding;
    int PADDED_W = W+2*padding;

    int a_in= Cin*H*W;
    int b_in= H*W;
    int c_in= W;
    int a_in_padded= Cin*PADDED_H*PADDED_W;
    int b_in_padded= PADDED_H*PADDED_W;
    int c_in_padded= PADDED_W;

    struct Image* PADDED_IN = (struct Image*)malloc(sizeof(struct Image));    
    
    int IN_size  = B*Cin*H*W;
    PADDED_IN->data = (float*)malloc(sizeof(float)*B*Cin*PADDED_H*PADDED_W);
    
    memset(PADDED_IN->data, 0.0, sizeof(float)*B*Cin*PADDED_H*PADDED_W);
    #pragma omp parallel for 
    for(int i=0; i< IN_size; i++)
    {
        PADDED_IN->data[a_in_padded*(i/a_in) + b_in_padded*((i/b_in)%Cin) + c_in_padded*((i/c_in)%c_in + padding) + i%c_in+padding] = IN->data[i];
    }
    
    free(IN->data);
    free(IN);
    //IN = NULL;


    //B*Cout*Hout*Wout;    
    int a_out= Cout*Hout*Wout;
    int b_out= Hout*Wout;
    int c_out= Wout;
   

    int out_size = B*a_out;
    int conv_filter_size = Cin*kernel_size*kernel_size;
    int b_conv = kernel_size*kernel_size;

    #pragma omp parallel for
    for(int i=0; i<out_size; i++)
    {
        float temp_conv =0.0;
        int Bout_index = i/a_out;
        int Cout_index = (i/b_out)%Cout;
        int Hout_index = (i/c_out)%c_out;
        int Wout_index = i%c_out;

        #pragma omp parallel for firstprivate( Bout_index, Cout_index, Hout_index, Wout_index)
        for (int j = 0; j < conv_filter_size; j++)
        {
            
            temp_conv+= weight[Cout_index*a+j]*(PADDED_IN->data[Bout_index*a_in_padded + ((j/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (j/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + j%kernel_size)]);
            if(i==0)
            {
            //printf("%d: weight: %f /  input: %f\n", j, weight[Cout_index*a+j], (PADDED_IN->data[Bout_index*a_in_padded + ((j/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (j/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + j%kernel_size)]));
            }
        }

        if(bias_exist==1)
        {
            temp_conv+=bias[Cout_index];
        }
        OUT->data[i] = temp_conv; 
        
    }
    //printf("%.10f\n", OUT->data[100]);
    //printf("CC\n");
    //printf("%.10f\n", OUT->data[100+Hout*Wout]);
    
    free(weight);
    free(weight_name);
    free(bias);
    free(bias_name);
    free(PADDED_IN->data);
    free(PADDED_IN);    
    
    return OUT;
}

struct Image * AveragePool(struct Image * IN, int kernel_size, int padding) //stride is set to 1;
{
    int B= IN->B;
    int C= IN->C;
    int H= IN->H+2*padding;
    int W= IN->W+2*padding;
    
    int INA = B*C;
    int Hout = (H-kernel_size+1);
    int Wout = (W-kernel_size+1);
    int OUTA = Hout* Wout;
    int K = kernel_size*kernel_size;
    int HW = H*W;
    int HWout = Hout*Wout; 
    int PADDED_IN_size = B*C*H*W;
    int IN_size = B*C*(H-2*padding)*(W-2*padding);

    struct Image * OUT = (struct Image*)malloc(sizeof(struct Image));
    OUT-> B = B;
    OUT-> C = C;
    OUT-> H = Hout;
    OUT-> W = Wout;
    OUT->data = (float*)malloc(sizeof(float)*OUTA*INA);


    if(padding!=0)
    {
        struct Image * PADDED_IN = (struct Image*)malloc(sizeof(struct Image));
        PADDED_IN->data = (float*)malloc(sizeof(float)*PADDED_IN_size);
        PADDED_IN->B = B;
        PADDED_IN->C = C;
        PADDED_IN->H = H;
        PADDED_IN->W = W;

        int a_in= C*(H-2*padding)*(W-2*padding);
        int b_in= (H-2*padding)*(W-2*padding);
        int c_in= W-2*padding;
        int a_in_padded= C*H*W;
        int b_in_padded= H*W;
        int c_in_padded= W;
     
        memset(PADDED_IN->data, 0.0, sizeof(float)*PADDED_IN_size);
        #pragma omp parallel for 
        for(int i=0; i< IN_size; i++)
        {
            PADDED_IN->data[a_in_padded*(i/a_in) + b_in_padded*((i/b_in)%C) + c_in_padded*((i/c_in)%c_in + padding) + i%c_in+padding] = IN->data[i];
        }

        free(IN->data);
        free(IN); 
        IN = PADDED_IN;
    }

    
    #pragma omp parallel for
    for(int i=0; i<INA; i++)
    {     
      
        //#pragma omp task firstprivate(i)
        {
            
            #pragma omp parallel for
            for(int j= 0; j<OUTA; j++)
            {
                float temp_sum =0.0;

              //  #pragma omp parallel for
                for(int k = 0 ; k< K; k++)
                {
                    temp_sum += IN->data[i*HW+ ((k/kernel_size)+j/(Wout))*W+((k%kernel_size)+j%Wout)];
                }

                OUT->data[i*HWout+j] = temp_sum/K;  
            }            

        }
        
    }

    free(IN->data);
    free(IN);

    return OUT;
}
struct Image * SoftMax(struct Image* IN)
{
    int B= IN->B;
    int C= IN->C;
    int BC = B*C;
    
    struct Image * OUT = (struct Image *)malloc(sizeof(struct Image));
    
    OUT->B =B;
    OUT->C =C;
    OUT->H =IN->H;
    OUT->W =IN->W;
    OUT->data = (float*)malloc(sizeof(float)*BC);

    //#pragma omp parallel for
    for(int i=0; i<B; i++)
    {
        float denom = 0.0;
        //#pragma omp parallel for firstprivate(i)
        for(int j=0; j<C; j++)
        {
            denom+= expf(IN->data[i*C+j]);                        
        }

        //#pragma omp parallel for firstprivate(i, denom)
        for(int j=0; j<C; j++)
        {
            OUT->data[i*C+j]= expf(IN->data[i*C+j])/denom;                        
        }

    }
    
    free(IN->data);
    free(IN);

    return OUT;
       
}
struct CMAX{
    float data;
    int index;
};
#pragma omp declare reduction(custom_max:CMAX: omp_out = (omp_in.data>omp_out.data)?omp_in:omp_out) initializer(omp_priv= {FLT_MIN, 0})

struct Image* ArgMax(struct Image* IN) // INPUT: (B, C) // OUTPUT: (B, 1) with its value as the index of which is max value
{
    int B = IN->B;
    int C = IN->C;
    //printf("ArgMax C is %d\n", C);
    struct Image* OUT = (struct Image*)malloc(sizeof(struct Image));
    OUT->B= B;
    OUT->C= 1;
    OUT->H= 1;
    OUT->W =1;
    OUT->data= (float*)malloc(sizeof(float)* B);
   /*
    #pragma omp parallel for
    for(int i=0; i<B; i++)
    {
        struct CMAX cmax;
        cmax.data =FLT_MIN;
        cmax.index = -5;

        #pragma omp parallel for firstprivate(i) reduction (custom_max: cmax)
        for(int j=0; j<C; j++)
        {
            struct CMAX ctemp;
            ctemp.data = IN->data[i*C+j];
            ctemp.index = j;
            if(cmax.data<ctemp.data)
            {
                cmax = ctemp;                
            }

        }
        OUT->data[i] =cmax.index;

    }
    */

    for(int i=0; i<B; i++)
    {
       
        float max_data =(float)(-INFINITY);
        //float max_data = -10000000.0;
        int max_index = -1;
        for(int j=0; j<C; j++)
        {
            
            float temp_data= IN->data[i*C+j];
           // printf("I is %d and data value is %f\n", i, temp_data);
            if(max_data<temp_data)
            {
                max_data = temp_data;
                max_index = j;                
            }

        }
        OUT->data[i] =max_index;

    }
     
    free(IN->data);
    free(IN);
    return  OUT;
}
struct Image * MaxPool(struct Image * IN, int kernel_size , int stride, int padding)
{
    int B= IN->B;
    int C= IN->C;
    int H= IN->H + 2*padding;
    int W= IN->W + 2*padding;
    
    int INA = B*C;
    int Hout = ((H-kernel_size)/stride)+1;
    int Wout = ((W-kernel_size)/stride)+1;
    int OUTA = Hout* Wout;
    int K = kernel_size*kernel_size;
    int HW = H*W;
    int HWout = Hout*Wout; 
    int PADDED_IN_size = B*C*H*W;
    int IN_size = B*C*(H-2*padding)*(W-2*padding);
    
    struct Image * OUT = (struct Image*)malloc(sizeof(struct Image));
    OUT-> B = B;
    OUT-> C = C;
    OUT-> H = Hout;
    OUT-> W = Wout;
    OUT->data = (float*)malloc(sizeof(float)*OUTA*INA);

    
    
    if(padding!=0)
    {
        struct Image * PADDED_IN = (struct Image*)malloc(sizeof(struct Image));
        PADDED_IN->data = (float*)malloc(sizeof(float)*PADDED_IN_size);
        PADDED_IN->B = B;
        PADDED_IN->C = C;
        PADDED_IN->H = H;
        PADDED_IN->W = W;

        int a_in= C*(H-2*padding)*(W-2*padding);
        int b_in= (H-2*padding)*(W-2*padding);
        int c_in= W-2*padding;
        int a_in_padded= C*H*W;
        int b_in_padded= H*W;
        int c_in_padded= W;
     
        //memset(PADDED_IN->data, -INFINITY, sizeof(float)*PADDED_IN_size);
        #pragma omp parallel for
        for(int i = 0; i< PADDED_IN_size; i++)
        {
            PADDED_IN->data[i]= (float)-INFINITY;
        }

       #pragma omp parallel for 
        for(int i=0; i< IN_size; i++)
        {
            PADDED_IN->data[a_in_padded*(i/a_in) + b_in_padded*((i/b_in)%C) + c_in_padded*((i/c_in)%c_in + padding) + i%c_in+padding] = IN->data[i];
        }

        free(IN->data);
        free(IN); 
        IN = PADDED_IN;
    }
    
   
    #pragma omp parallel for
    for(int i=0; i<INA; i++)
    {     
      
        //#pragma omp task firstprivate(i)
        {
            
            #pragma omp parallel for
            for(int j= 0; j<OUTA; j++)
            {
                float temp_max =(float)(-INFINITY);

               #pragma omp parallel for reduction(max: temp_max)
                for(int k = 0 ; k< K; k++)
                {
                    if(temp_max< IN->data[i*HW+((j/Wout*stride)+(k/kernel_size))*W+(((j%Wout)*stride)+(k%kernel_size))]);
                    {
                        temp_max =  IN->data[i*HW+((j/Wout*stride)+(k/kernel_size))*W+(((j%Wout)*stride)+(k%kernel_size))];
                    }
                }

                OUT->data[i*HWout+j] = temp_max;  
            }            

        }
        
    }        

    free(IN->data);
    free(IN); 
   
    return OUT;
}

struct Image* Linear(struct Image * IN, int Cin, int Cout, cJSON * cjsonp, const char* name)
{
    int B = IN->B;    
    int H = IN->H;
    int W = IN->W;

    struct Image* OUT = (struct Image*)malloc(sizeof(struct Image));
    OUT->B =B;
    OUT->C =Cout;
    OUT->H =H;
    OUT->W =W;
    OUT->data = (float*)malloc(sizeof(float)*B*Cout*H*W);
    char * weight_name= (char*)malloc(sizeof(char)*(50));
    char * bias_name = (char*)malloc(sizeof(char)*(50));

    int weight_size = Cin* Cout;
    int bias_size= Cout;
    int bias_exist = 0;
    float * weight = (float*)malloc(sizeof(float)*weight_size);
    float * bias =  (float*)malloc(sizeof(float)*bias_size);

    //////WEIGHT INITIALIZATION
    strcpy(weight_name, name);
    strcat(weight_name, ".weight");
    cJSON * layer_weight = cJSON_GetObjectItem(cjsonp, weight_name);
    if((cJSON_GetArraySize(layer_weight)!=Cout)||(cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0))!=Cin))
    {
        printf("Current Weight Matrix size is  %d x %d, But Real Weight Matrix size is %d x %d!\n", cJSON_GetArraySize(layer_weight), cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0)), Cout, Cin);
        printf("Size does not match!\n");
        return NULL;
    }
    #pragma omp parallel for
    for(int i=0; i<weight_size; i++)
    {
        cJSON* value = cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, i/Cin), i%Cin);
        weight[i] = (float)(value->valuedouble);
    }

    //////BIAS INITIALIZATION
    strcpy(bias_name, name);
    strcat(bias_name, ".bias");
    cJSON * layer_bias = cJSON_GetObjectItem(cjsonp, bias_name);
    if(layer_bias != NULL)
    {
        bias_exist =1;
        if(cJSON_GetArraySize(layer_bias)!=Cout)
        {
            printf("Current Bias size is  %d , But Real Bias size is %d!\n", cJSON_GetArraySize(layer_bias), Cout);
            printf("Size does not match!\n");
            return NULL;
        }
        #pragma omp parallel for
        for(int i=0; i<bias_size; i++)
        {
            cJSON* value = cJSON_GetArrayItem(layer_bias, i);
            bias[i] = (float)(value->valuedouble);
        }

    }

    #pragma omp parallel for 
    for(int i =0; i< B*Cout; i++)
    {
        float result = 0.0;
       
        #pragma omp parallel for
        for(int j=0; j< Cin; j++)
        {
            result+= IN->data[Cin*(i/Cout)+j]*weight[(i%Cout)*Cin+j];
        }
        if(bias_exist==1)
        {
            result+= bias[i%Cout];
        }            
           
        
        OUT->data[i] = result;
    }
     
    free(weight);
    free(weight_name);
    free(bias);
    free(bias_name);
    free(IN->data);
    free(IN);
    return OUT;

}

struct Image * ReLU(struct Image * IN)
{
    int size = IN->B*IN->H*IN->W*IN->C;
    //printf("The size is %d\n",size);
    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i< size; i++)
        {
            if(IN->data[i]<0)
            {
                IN->data[i] = 0.0;
            }
        }
    }
    return IN;
}
struct Image * Batchnorm(struct Image * IN, cJSON * cjsonp, const char* name,  float eps)
{
    int B = IN->B;
    int C = IN->C;
    int H = IN->H;
    int W = IN->W;
    //printf("batch%d, %d, %d, %d\n", B,C,H,W);
    
    int whole = B*H*W;
    int size =  sizeof(IN->data)/sizeof(float);
    int x = H*W*C;
    int y = H*W;
    int z = W;

    float * mean = (float*)malloc(C* sizeof(float)); 
    float * rstddev = (float*)malloc(C* sizeof(float));
    float * gamma = (float*)malloc(C* sizeof(float)); 
    float * beta = (float*)malloc(C* sizeof(float));
    struct Image * OUT = (struct Image*)malloc(sizeof(struct Image));
    OUT->data = (float*)malloc(sizeof(float)*whole*C);
    if(OUT->data == NULL)
    {
        printf("Not Allocated!\n");
    }
    OUT->B = B;
    OUT->H = H;
    OUT->W = W;
    OUT->C = C;
    char * weight_name= (char*)malloc(sizeof(char)*(50));
    char * bias_name = (char*)malloc(sizeof(char)*(50));   

    //printf("Flag0!\n");
    //printf("%d", OUT->data[0]);


    //////GAMMA INITIALIZATION
    strcpy(weight_name, name);
    strcat(weight_name, ".weight");
    cJSON * layer_weight = cJSON_GetObjectItem(cjsonp, weight_name);
    //cJSON_Print(layer_weight);
    

    if(cJSON_GetArraySize(layer_weight)!=C)
    {
        printf("Size is %d and C size is %d\n",cJSON_GetArraySize(layer_weight), C);
        printf("Size does not match!\n");
        return NULL;
    }
    #pragma omp parallel for
    for(int i=0; i<C; i++)
    {
        cJSON* value = cJSON_GetArrayItem(layer_weight, i);
        gamma[i] = (float)(value->valuedouble);
    }

    /////BETA INITIALIZATION
    strcpy(bias_name, name);
    strcat(bias_name, ".bias"); 
    cJSON * layer_bias = cJSON_GetObjectItem(cjsonp, bias_name);
    if(cJSON_GetArraySize(layer_bias)!=C)
    {
        printf("Size is %d and C size is %d\n",cJSON_GetArraySize(layer_bias), C);
        printf("Size does not match!\n");
        return NULL;
    }
    #pragma omp parallel for 
    for(int i=0; i<C; i++)
    {
        cJSON* value = cJSON_GetArrayItem(layer_bias, i);
        beta[i] = (float)(value->valuedouble);
    }   
    //printf("Process2 done!\n");
 
    #pragma omp parallel 
    
        #pragma omp for
        for(int i =0; i<C; i++)
        {
           #pragma omp task firstprivate(i)
            {
                float sum_mean = 0.0;
                float sum_mean_sq = 0.0;

                #pragma omp parallel for 
                for(int index=0; index< whole; index++)                
                {
                    //printf("%.10f\n", IN->data[index%y+(index/y)*x+i*y]);
                    //printf("BB\n");
                    //printf("%.10f\n", IN->data[index%y+(index/y)*x+i*y+y]);
                    sum_mean+=IN->data[(index%y)+(index/y)*x+i*y];
                    sum_mean_sq+=IN->data[index%y+(index/y)*x+i*y]*IN->data[index%y+(index/y)*x+i*y];
                }                   
                //printf("%.10f\n", sum_mean);
                mean[i]= sum_mean/ whole;
               
                rstddev[i]=sqrtf(((sum_mean_sq/whole) - (mean[i]*mean[i]))+eps); 
                //rstddev[i]= sqrtf(rstddev[i]);               
                
            }
        }
        
        //printf("\n");
    
       //printf("Process3 done!\n");
    
    #pragma omp parallel
    {
        #pragma omp  parallel for 
        for(int i=0; i<C; i++)
        {
            #pragma omp task firstprivate(i)
            {
                float temp_gamma = gamma[i];

                float temp_beta = beta[i];
                float temp_mean = mean[i];
                float temp_rstddev =  rstddev[i];
               // printf("%f\n", temp_mean);
                // printf("%f\n", temp_rstddev);
                // printf("%f\n", temp_gamma);
                // printf("%f\n", temp_beta);
                #pragma omp parallel for 
                for(int index=0; index<whole; index++)
                {
                    //printf("Index is %d\n",index);
                    //printf("Cal is %d\n",index%y+(index/y)*x+i*y);
                    
                    OUT->data[index%y+(index/y)*x+i*y] = temp_gamma*((IN->data[index%y+(index/y)*x+i*y]-temp_mean)/(temp_rstddev))+temp_beta;
                }

            }
            
        }
    }    
     //printf("Process4 done!\n");
    free(mean);
    free(rstddev);
    free(gamma);
    free(beta);
    free(weight_name);
    free(bias_name);
    free(IN->data);
    free(IN); 
    return OUT;
}

float randfloat(float a, float b) {
    srand(time(NULL));
    return a + (float)rand() / RAND_MAX * (b - a);
}

struct Image* ADD(struct Image* IN0, struct Image* IN1)
{
    struct Image* OUT = (struct Image*)malloc(sizeof(struct Image));
    int  IN_size = (IN0->B)*(IN0->C)*(IN0->H)*(IN0->W);

    OUT->B =IN0->B;
    OUT->C =IN0->C;
    OUT->H =IN0->H;
    OUT->W =IN0->W;
    OUT->data = (float*)malloc(sizeof(float)*IN_size);

    #pragma omp parallel for
    for(int i=0; i<IN_size; i++)
    {
        OUT->data[i] = IN0->data[i] + IN1->data[i];
    }

    free(IN0->data);
    free(IN1->data);
    free(IN0);
    free(IN1);
    return OUT;
}

struct Image* Block(struct Image* IN, int Cin, int Cout, int downsample, cJSON* cjsonp,const char* name)
{    
    char  conv0_name[50];
    char  bn0_name[50];
    char  conv1_name[50];
    char  bn1_name[50];
    char  ds_conv0_name[50];
    char  ds_bn0_name[50];
    float eps = 0.00001;

    strcpy(conv0_name, name);
    strcpy(bn0_name, name);
    strcpy(conv1_name, name);
    strcpy(bn1_name, name);
    strcpy(ds_conv0_name, name);
    strcpy(ds_bn0_name, name);


    strcat(conv0_name, ".conv1");
    strcat(bn0_name, ".bn1");
    strcat(conv1_name, ".conv2");
    strcat(bn1_name, ".bn2");
    strcat(ds_conv0_name, ".downsample.0");
    strcat(ds_bn0_name, ".downsample.1");


    struct Image * IN1 = (struct Image*)malloc(sizeof(struct Image));
    int IN_size = (IN->B)*(IN->C)*(IN->H)*(IN->W);
    IN1->data= (float*)malloc(sizeof(float)*IN_size);
    IN1->B = IN->B;
    IN1->C = IN->C;
    IN1->H = IN->H;
    IN1->W = IN->W;

    #pragma omp parallel for
    for(int i=0; i< IN_size; i++)
    {
        IN1->data[i] = IN->data[i];
    }    
   // printf("AA0\n");
    IN = Conv(IN, Cin, Cout, 3, (downsample==1)?2:1,1, cjsonp, conv0_name);    
   // printf("AA1\n");
    
    IN = Batchnorm(IN, cjsonp, bn0_name, eps);    
  //  printf("AA2\n");
    IN = ReLU(IN);
   // printf("AA3\n");
    IN = Conv(IN, Cout, Cout, 3, 1,1, cjsonp, conv1_name);
   // printf("AA4\n");
    IN = Batchnorm(IN, cjsonp, bn1_name, eps); 
    //printf("AA5\n");
    return ReLU((downsample==0)?ADD(IN, IN1):ADD(IN,Batchnorm(Conv( IN1, Cin ,Cout, 1, 2, 0, cjsonp, ds_conv0_name),cjsonp, ds_bn0_name, eps)));
    
    return IN;
}
struct Image* ForwardResnet18(struct Image* IN, cJSON* cjsonp, int num)
{
    float eps = 0.00001;
   // if(num==20) printf("A0\n");
    //printf("A0\n");
    //printf("%d", IN->B);
    IN = Conv(IN, 3, 64, 7, 2, 3, cjsonp, "conv1");
    //printf("A0\n");
    //if(num==20) printf("A1\n");
    //printf("AA1\n") ;   
     IN = Batchnorm(IN, cjsonp, "bn1", eps);  
   // printf("AA1\n");  
    IN = ReLU(IN);  
    //printf("IN->HB0%d\n",IN->H);
    IN = MaxPool(IN, 3, 2, 1);
   // printf("A0\n");
    IN = Block(IN, 64, 64 , 0, cjsonp, "layer1.0");
   // if(num==20) printf("A2\n");
    IN = Block(IN, 64, 64 , 0, cjsonp, "layer1.1");
    //printf("IN->HB1%d\n",IN->H);
   // printf("A1\n");
     IN = Block(IN, 64, 128 , 1, cjsonp, "layer2.0");
    //printf("F%d\n",IN->H);
    //if(num==20) printf("A3\n");
   IN = Block(IN, 128, 128 , 0, cjsonp, "layer2.1");
   // printf("IN->HB2%d\n",IN->H);
    //printf("A2\n");
    IN = Block(IN, 128, 256 , 1, cjsonp, "layer3.0");
    IN = Block(IN, 256, 256 , 0, cjsonp, "layer3.1");
    //printf("IN->HB3%d\n",IN->H);
    //printf("A3\n");
  // if(num==20) printf("A4\n");
    IN = Block(IN, 256, 512 , 1, cjsonp, "layer4.0");
    IN = Block(IN, 512, 512 , 0, cjsonp, "layer4.1");
    //if(num==20) printf("A5\n");
   // printf("IN->HB4%d\n",IN->H);
    //printf("A4\n");
    IN = AveragePool(IN, 7, 0);
    //if(num==20) printf("A6\n");
    //printf("IN->H%d\n",IN->H);
     IN = Linear(IN, 512, 100, cjsonp, "fc");
    //IN = SoftMax(IN);
   // printf("A5\n");
   //if(num==20) printf("A7\n");
    IN = ArgMax(IN);
    //printf("A6\n");
    //if(num==20) printf("A8\n");
    return IN;
}




struct Limage
{
    float clabel;
    float flabel;
    //unsigned char raw_image[3072];     
};

struct Dataloader
{    
    struct Image* resized_images;
    unsigned char* clabel;
    unsigned char* flabel;
    unsigned char* raw_image;   
    int batch_size;
    int total_image;
    int shuffle;
    int* indices;
    int current_index;
};

void Readbatch(const char * name, struct  Dataloader * dataloader, long int* position)
{
    FILE* file = fopen(name, "rb");
    if (!file) 
    {
        printf("Error: Cannot open file %s\n", name);
        return;
    }
    fseek(file, *position, SEEK_SET);

    int start_index = dataloader->current_index;
    int end_index = start_index + dataloader->batch_size;
    if (end_index > dataloader->total_image) 
    {
        end_index = dataloader->total_image;
    }

    int current_batch_size = end_index - start_index;
    

    dataloader->clabel = (unsigned char *)malloc(current_batch_size*sizeof(unsigned char));
    dataloader->flabel = (unsigned char *)malloc(current_batch_size*sizeof(unsigned char));
    dataloader->raw_image = (unsigned char *)malloc(current_batch_size*3072*sizeof(unsigned char));
    if (!dataloader->raw_image) 
    {
        printf("Error: Unable to allocate memory for the dataset\n");
        return;
    }
    

    for (int i = 0; i < current_batch_size; i++) 
    {
        
        fread(dataloader->clabel+i, 1, 1, file);
        fread(dataloader->flabel+i, 1, 1, file);      
        fread(dataloader->raw_image+3072*i, 3072, 1, file);    
    }
    //printf("0th sample value is %u\n", dataloader->raw_image[0]);
    *position = ftell(file);
    fclose(file);

}
void Transform(struct Image** rimages, unsigned char* raw_image, int batch_size, int image_size, int current_index)
{
    int whole_size = 3*batch_size*image_size*image_size;
    int size_wo_batch = 3*image_size*image_size;
   // printf("A-1\n");
    
    (*rimages) = (struct Image*)malloc(sizeof(struct Image));
   // printf("A-2\n");
    (*rimages)->data = (float*)malloc(sizeof(float)*whole_size);
   // printf("A-3\n");
    (*rimages)->B =batch_size;
    (*rimages)->C =3;
    (*rimages)->H =image_size;
    (*rimages)->W =image_size;
   // printf("A-5\n");
    //printf("whole size is %d\n", whole_size);
   // printf("%f\n", (*rimages)->data[302072]);
   // printf("A-6\n");
    //RESIZE
    //printf("A0\n");
    float ratio = (image_size) /((float)(32));
    int b =image_size*image_size;
    int b_raw =32*32;
   
    // printf("A00\n");
    #pragma omp parallel for
    for (int i=0; i< batch_size; i++)
    {                
       //printf(" i is %d\n",i);
        
        int ii= 0;
        int jj= 0;
        for (int j = 0; j<size_wo_batch; j++) 
        {
            int c= (j/b)%3;
            int h_index= (j/image_size)%image_size;
            int w_index= j%image_size;
            float aa= ratio*ii+3;
            float bb= ratio*jj+3;
            if(w_index ==0)
            {
                jj= 0;
            }
            if(h_index==0)
            {
                ii=0;
            }
            
            if(h_index< aa)
            {

            }
            else
            {
                ii++;
                aa+=ratio;                
            }

            if(w_index< bb)
            {

            }
            else
            {
                jj++;
                bb+=ratio;                
            }
            int h1= ii-1;
            int h2= ii;
            int w1= jj-1;
            int w2= jj;
            if(h1<0)
            {
                h1=0;
            }
             if(w1<0)
            {
                w1=0;
            }
             if(h2>31)
            {
                h2=31;
            }
             if(w2>31)
            {
                w2=31;
            }


            //float src_h= h_index*ratio;
            //float src_w= w_index*ratio;

           // int h1 = (int)src_h;
           // int w1 = (int)src_w;
            

           // int h2= (h1+1 < 32)? h1+1: h1;
            //int w2= (w1+1 < 32)? w1+1: w1;

            float dh = (float)(h_index)-(aa-ratio);
            float dw = (float)(w_index)-(bb-ratio);
            //if(dh==0.0)      printf("dh: %.10f",dh);
            //float dw = src_w - w1;
            //int a0= c*b_raw+h1*32+w1;
           //int a1= c*b_raw+h1*32+w2;
            //int a2= c*b_raw+h2*32+w1;
            //int a3= c*b_raw+h2*32+w2;           

            unsigned char p1 = raw_image[i*b_raw*3+c*b_raw+h1*32+w1];
            unsigned char p2 = raw_image[i*b_raw*3+c*b_raw+h1*32+w2];
            unsigned char p3 = raw_image[i*b_raw*3+c*b_raw+h2*32+w1];
            unsigned char p4 = raw_image[i*b_raw*3+c*b_raw+h2*32+w2];
                
            int pixel_value = round(((ratio-dw)/ratio)*((ratio-dh)/ratio)*p1 +(dw/ratio)*((ratio-dh)/ratio)*p2 + ((ratio- dw)/ratio)*(dh/ratio)*p3 + (dw/ratio)*(dh/ratio)*p4);
            if((j==1)||(j==0))
            {
                //printf("AAAA\n");
               // printf("%f\n",dw);
               // printf("%u\n",p1);
               // printf("%u\n",p2);
               // printf("%f\n", ((ratio-dw)/ratio)*((ratio-dh)/ratio)*p1+(dw/ratio)*((ratio-dh)/ratio)*p2);
            }
            //float pixel_value = raw_image[i*b_raw*3+c*b_raw+h1*32+w1];
            if((pixel_value<0)||(pixel_value>255))
            {
                printf("%d\n", pixel_value);
            }
            //printf("A2\n");
            
            //printf(" i is %d , j is%d\n", i,j);
            //printf("%f\n", pixel_value);
            (*rimages)->data[i*b*3+j] = (float)pixel_value;
            //printf("%f\n", pixel_value);
            //printf("A3\n"); 

        }

    }
    //printf("A1\n");
    //TOTensor
    #pragma omp parallel for
    for (int i=0; i<batch_size; i++)
    {
        #pragma omp parallel for firstprivate(i)
        for(int j=0; j< size_wo_batch;j++)
        {
            (*rimages)->data[i*b*3+j]= ((float)((*rimages)->data[i*b*3+j]))/((float)(255.0));
        }

    }
    
    //Normalize
    float mean[3] = {0.5071, 0.4867, 0.4408};
    float std[3]  = {0.2675, 0.2565, 0.2761};
    #pragma omp parallel for
    for (int i=0; i< batch_size; i++)
    {
        #pragma omp parallel for firstprivate(i)
        for(int j=0; j< size_wo_batch;j++)
        {
            int c = (j/b)%3;
            (*rimages)->data[i*b*3+j]= ((*rimages)->data[i*b*3+j]-(mean[c]))/(std[c]);
        }
       
    }
    

    return;

}
////////////////////////////BFP///////////////////////////

struct bfp_data float2bfp(float data, int mant_bit_size)
{
    struct bfp_data bfp;
    int temp;
 
    if(data==0.0)
    {
      bfp.sign=false;	    
      bfp.exp=0;
      bfp.mant=0;
      return bfp;

    }
 

    uint32_t* uint32_data=  (uint32_t *)&data;
    bfp.sign= ((*uint32_data >> 31) ==1)?true:false;
    bfp.exp=  (*uint32_data >> 23) & 0xFF;
    temp = (*uint32_data) & 0x7FFFFF;
    uint32_t mask = (1<<(24-mant_bit_size)) -1;
    uint32_t aa = temp & mask;
    bfp.mant= (bfp.exp==0)?(temp>>(23-mant_bit_size)):(temp|(1<<23))>> (24-mant_bit_size);

    double random = (double)rand() / RAND_MAX;

    double p= (double)aa/(1<<(24-mant_bit_size));
    if(random<p)
    {
        bfp.mant +=1;
    }
    
    return bfp;
}

float bfp2float(struct bfp_data data, int mant_bit_size)
{
    float result;

    result= (data.sign==true?-1:1)*powf(2.0, data.exp-127-(mant_bit_size-1))* data.mant;

    return result;
}


float bfp2float_aftermul(struct bfp_data data, int mant_bit_size)
{
    float result;

    result= (data.sign==true?-1:1)*powf(2.0, data.exp-2*(127+(mant_bit_size-1)))* data.mant;

    return result;
}



struct bfp_data bfpmul(struct BFP data1, struct BFP data2, int mant_bit_size)
{
    struct bfp_data result= {0};
    result.exp= data1.max_exp + data2.max_exp;
    int sum=0;
    //#pragma omp parallel for
    for(int i=0; i<data1.g; i++)
    {
        sum+= ((data1.sign_array[i]!=data2.sign_array[i])?-1:1)*(data1.mant_array[i])*(data2.mant_array[i]);
    }
    result.sign= (sum<0)?true:false;
    //sum= sum>>(mant_bit_size-1);
    sum= abs(sum);
    result.mant= sum;
   //printf("Sum is %d\n", sum);
    int loc = 31 - __builtin_clz(sum);

    if(loc==-1)
    {
        return result;
    }
    
    //int loc_diff = loc - 2*(mant_bit_size-1);
    //printf("loc diff value is %d\n",loc_diff);
    //result.exp+=loc_diff;

    //result.mant = (loc_diff>0)?(result.mant >> loc_diff):(result.mant<<(-loc_diff));
    //printf("mant is %d\n", result.mant);
    //result.mant = result.mant>>(mant_bit_size-1);
    
    //result.exp+=(mant_bit_size-1);
    //printf("mant is %d\n", result.mant);
    return result;
}

struct bfp_data bfpadd(struct bfp_data data1, struct bfp_data data2, int mant_bit_size)
{
    struct bfp_data result= {0};
    if(data1.exp==0)
    {
        if(data2.exp==0)
        {
            return result;
        }
        else
        {
            return data2;
        }

    }
    else
    {
        if(data2.exp==0)
        {
            return data1;
        }
    }

    int max_exp = max(data1.exp, data2.exp);
    int min_exp = min(data1.exp, data2.exp);
    result.exp = max_exp;
    int diff = max_exp - min_exp;
    
    if(data1.exp>data2.exp)
    {
       data2.mant= data2.mant >> diff;
    }
    else
    {
       data1.mant= data1.mant >> diff;
    }
    
    result.mant= data1.mant*((data1.sign==true)?-1:1)  + data2.mant*((data2.sign==true)?-1:1);
    
    result.sign= (result.mant>0)?false:true;
   
    
    result.mant= abs(result.mant);
    int loc = 31 - __builtin_clz(result.mant);
    if(loc==-1)
    {
        return result;
    }
    int loc_diff = loc - (mant_bit_size-1);
   
    result.exp+=loc_diff;
    
    result.mant = (loc_diff>0)?(result.mant >> loc_diff):(result.mant<<(-loc_diff));
    
    

    return result;
}
void form_group(int x, int y, int z, struct bfp_data* p, int group_size, int kernel_size, struct bfp_data *out)
{
    #pragma omp parallel for
    for(int i=0; i<group_size; i++)
    {
        (out)[i]= p[(group_size*z+i)*kernel_size*kernel_size + x*kernel_size + y];
    }
    return;
}

void form_group_basic(int j, struct bfp_data* p, int group_size, int kernel_size, struct bfp_data *out)
{
    int barrier= 3*kernel_size*kernel_size;

    #pragma omp parallel for
    for(int i=0; i<group_size; i++)
    {
        (out)[i]={0};
        int index= j*group_size+i;
        if(index< barrier)
        {
            (out)[i]= p[index];
        }
        
    }
    return;
}

struct BFP  form_BFP(struct bfp_data* p, int group_size)
{
    struct BFP result;
    
    //result.mant_array= (int*)malloc(sizeof(int)*group_size); 
    //result.sign_array= (bool*)malloc(sizeof(bool)*group_size); 
    int temp_max_exp= -1000;
    int temp=0;
    //#pragma omp parallel for
    for(int i=0; i<group_size; i++)
    {  
        if(temp_max_exp<p[i].exp)
        {
            temp_max_exp= p[i].exp;
        }
    }
    result.g=group_size;
    result.max_exp=temp_max_exp;

    #pragma omp parallel for
    for(int i=0; i<group_size; i++)
    {
        result.sign_array[i]=p[i].sign;
        int diff =  temp_max_exp- p[i].exp;
        result.mant_array[i] = p[i].mant >> diff;
        //temp++;
    }   

    return result;
}

void form_group_linear(int index, struct bfp_data* p, int group_size, struct bfp_data** out)
{
    #pragma omp parallel for
    for(int i=0; i<group_size; i++)
    {
        (*out)[i]= p[index*group_size+i];
    }
    return;
}


struct BFP form_BFP_direct_linear_w(struct bfp_data* weight, int outer_index, int tile_index, int Cin, int Cout)
{
    struct BFP result;
    int temp_max_exp= -1000;

    int k=(outer_index%Cout)*Cin+16*tile_index;

    //#pragma omp parallel for reduction(max:temp_max_exp)
    for(int i=0; i<16; i++)
    {
        if(temp_max_exp<weight[k+i].exp)
        {
            temp_max_exp= weight[k+i].exp;
        }
    }

    result.g=16;
    result.max_exp=temp_max_exp;

    //#pragma omp parallel for
    for(int i=0; i<16; i++)
    {
        result.sign_array[i]=weight[k+i].sign;
        int diff = temp_max_exp - weight[k+i].exp;
        result.mant_array[i] = weight[k+i].mant >> diff;
    }

    return result;
}

struct BFP form_BFP_direct_linear_i(struct bfp_data* input, int outer_index, int tile_index, int Cin, int Cout)
{
    struct BFP result;
    int temp_max_exp= -1000;

    int k=Cin*(outer_index/Cout)+16*tile_index;

    //#pragma omp parallel for reduction(max:temp_max_exp)
    for(int i=0; i<16; i++)
    {
        if(temp_max_exp<input[k+i].exp)
        {
            temp_max_exp= input[k+i].exp;
        }
    }

    result.g=16;
    result.max_exp=temp_max_exp;

    //#pragma omp parallel for
    for(int i=0; i<16; i++)
    {
        result.sign_array[i]=input[k+i].sign;
        int diff = temp_max_exp - input[k+i].exp;
        result.mant_array[i] = input[k+i].mant >> diff;
    }

    return result;
}

struct BFPImage* BFPLinear(struct BFPImage * IN, int Cin, int Cout, int mant_bit_size, int group_size, cJSON * cjsonp, cJSON* cjsonp2, const char* name)
{
    int B = IN->B;    
    int H = IN->H;
    int W = IN->W;

    struct BFPImage* OUT = (struct BFPImage*)malloc(sizeof(struct BFPImage));
    OUT->B =B;
    OUT->C =Cout;
    OUT->H =H;
    OUT->W =W;
    OUT->data = (struct bfp_data *)malloc(sizeof(struct bfp_data)*B*Cout*H*W);
    char * weight_name= (char*)malloc(sizeof(char)*(50));
    char * bias_name = (char*)malloc(sizeof(char)*(50));

    int weight_size = Cin*Cout;
    int bias_size= Cout;
    int bias_exist = 0;
    struct bfp_data * weight = (struct bfp_data*)malloc(sizeof(struct bfp_data)*weight_size);
    struct bfp_data * bias =  (struct bfp_data*)malloc(sizeof(struct bfp_data)*bias_size);

    //////WEIGHT INITIALIZATION
    strcpy(weight_name, name);
    strcat(weight_name, ".weight");
   
    //cJSON *item = cJSON_GetObjectItemCaseSensitive(cjsonp2, weight_name);
    //mant_bit_size = ((item)->valueint)-8-1;

    cJSON * layer_weight = cJSON_GetObjectItem(cjsonp, weight_name);
    if((cJSON_GetArraySize(layer_weight)!=Cout)||(cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0))!=Cin))
    {
        printf("Current Weight Matrix size is  %d x %d, But Real Weight Matrix size is %d x %d!\n", cJSON_GetArraySize(layer_weight), cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0)), Cout, Cin);
        printf("Size does not match!\n");
        return NULL;
    }
    #pragma omp parallel for
    for(int i=0; i<weight_size; i++)
    {
        cJSON* value = cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, i/Cin), i%Cin);
        weight[i] = float2bfp((float)(value->valuedouble),mant_bit_size);
    }


     //////BIAS INITIALIZATION
    strcpy(bias_name, name);
    strcat(bias_name, ".bias");
    cJSON * layer_bias = cJSON_GetObjectItem(cjsonp, bias_name);
    if(layer_bias != NULL)
    {
        bias_exist =1;
        if(cJSON_GetArraySize(layer_bias)!=Cout)
        {
            printf("Current Bias size is  %d , But Real Bias size is %d!\n", cJSON_GetArraySize(layer_bias), Cout);
            printf("Size does not match!\n");
            return NULL;
        }
        #pragma omp parallel for
        for(int i=0; i<bias_size; i++)
        {
            cJSON* value = cJSON_GetArrayItem(layer_bias, i);
            bias[i] = float2bfp((float)(value->valuedouble),mant_bit_size);
        }

    }

    //Linear Operation
    int num_tile = Cin/group_size;
    int intermediate_sum_bit_size= 20;
    
    #pragma omp parallel for
    for(int i =0; i< B*Cout; i++)
    {
        //float result = 0.0;
        //struct bfp_data temp_sum={0};

	float temp_sum=0.0;    
        //struct bfp_data* weight_chunk = (struct bfp_data*)malloc(sizeof(struct bfp_data)*Cin);
        //struct bfp_data* input_chunk = (struct bfp_data*)malloc(sizeof(struct bfp_data)*Cin);
        //printf("BFProcess11!\n");
        /*#pragma omp parallel for
        for(int j=0; j< Cin; j++)
        {
            weight_chunk[j]=weight[(i%Cout)*Cin+j];
            input_chunk[j]=IN->data[Cin*(i/Cout)+j];
        }*/

        #pragma omp parallel for
        for (int j=0; j< num_tile; j++)
        {
           // struct bfp_data* temp_w= (struct bfp_data*)malloc(sizeof(struct bfp_data)*group_size);
           // struct bfp_data* temp_i=(struct bfp_data*)malloc(sizeof(struct bfp_data)*group_size);
           // form_group_linear(j, weight_chunk,group_size, &temp_w);
           // form_group_linear(j,input_chunk,group_size, &temp_i);
            struct BFP BFP_w = form_BFP_direct_linear_w(weight, i, j, Cin, Cout);
            struct BFP BFP_i = form_BFP_direct_linear_i(IN->data, i, j, Cin, Cout);
            
            struct bfp_data result = bfpmul(BFP_w, BFP_i, mant_bit_size);
            //temp_sum= bfpadd(temp_sum, result, intermediate_sum_bit_size);

	    temp_sum+=bfp2float_aftermul(result,mant_bit_size);
            
           // free(temp_w);
           // free(temp_i);
           // free(BFP_w.mant_array);
            //free(BFP_i.mant_array);
            //free(BFP_w.sign_array);
          //  free(BFP_i.sign_array);

        }
        //float a1 = bfp2float_aftermul(temp_sum, mant_bit_size);        
        //temp_sum= float2bfp(a1, mant_bit_size);

        /*for(int j=0; j< Cin; j++)
        {
            result+= IN->data[Cin*(i/Cout)+j]*weight[(i%Cout)*Cin+j];
        }*/
        if(bias_exist==1)
        {
            //result+= bias[i%Cout];
            temp_sum+=bfp2float(bias[i%Cout], mant_bit_size);
	    //temp_sum= bfpadd(temp_sum, bias[i%Cout], mant_bit_size);
        }            
           
        
        OUT->data[i] = float2bfp(temp_sum, mant_bit_size);
        //free(weight_chunk);
        //free(input_chunk);

    }
     
    free(weight);
    free(weight_name);
    free(bias);
    free(bias_name);
    free(IN->data);
    free(IN);
    return OUT;
}


struct Image* BFPImage2Image(struct BFPImage* IN, int mant_bit_size)
{
    struct Image *OUT = (struct Image *)malloc(sizeof(struct Image));
    int whole= (IN->B)*(IN->C)*(IN->H)*(IN->W);
    OUT->data= (float*)malloc(sizeof(float)*whole);
    OUT->B = IN->B;
    OUT->C = IN->C;
    OUT->H = IN->H;
    OUT->W = IN->W;
    //#pragma omp parallel for
    for(int i=0; i<whole; i++)
    {
        OUT->data[i]= bfp2float(IN->data[i], mant_bit_size);
    }
    free(IN->data);
    free(IN);
    return OUT; 

}

struct BFPImage* Image2BFPImage(struct Image* IN, int mant_bit_size)
{
    struct BFPImage *OUT = (struct BFPImage *)malloc(sizeof(struct BFPImage));
    int whole= (IN->B)*(IN->C)*(IN->H)*(IN->W);
    OUT->data= (struct bfp_data*)malloc(sizeof(struct bfp_data)*whole);
    OUT->B = IN->B;
    OUT->C = IN->C;
    OUT->H = IN->H;
    OUT->W = IN->W;
    //#pragma omp parallel for
    for(int i=0; i<whole; i++)
    {
        OUT->data[i]= float2bfp(IN->data[i], mant_bit_size);
    }
    free(IN->data);
    free(IN);
    return OUT; 

}

struct BFP form_BFP_direct_w(struct bfp_data* weight, int Cin, int Cout_index, int conv_filter_size, int j , int isfirst, int kernel_size, int x, int y, int z)
{
    struct BFP result;
    int temp_max_exp= -1000;

    int barrier= Cin*kernel_size*kernel_size;
    if(isfirst==1)
    {
        for(int i=0; i<16; i++)
        {  
            if(j*16+i<barrier)
            {
                if(temp_max_exp<weight[Cout_index*conv_filter_size+j*16+i].exp)
                {
                    temp_max_exp= weight[Cout_index*conv_filter_size+j*16+i].exp;
                }
            }
            
        }
    }
    else
    {
        for(int i=0; i<16; i++)
        {  
           
            if(temp_max_exp<weight[Cout_index*conv_filter_size+(16*z+i)*kernel_size*kernel_size + x*kernel_size + y].exp)
            {
                temp_max_exp= weight[Cout_index*conv_filter_size+(16*z+i)*kernel_size*kernel_size + x*kernel_size + y].exp;
            }
            
            
        }
    }
    result.g=16;
    result.max_exp=temp_max_exp;  
    
    if(isfirst==1)
    {
        //#pragma omp parallel for
        for(int i=0; i<16; i++)
        {
            if(j*16+i<barrier)
            {
                result.sign_array[i]=weight[Cout_index*conv_filter_size+j*16+i].sign;
                int diff =  temp_max_exp- weight[Cout_index*conv_filter_size+j*16+i].exp;
                result.mant_array[i] = weight[Cout_index*conv_filter_size+j*16+i].mant >> diff;
            }
            else
            {
                result.sign_array[i]=0;
                result.mant_array[i]=0;
            }        

        }
    }
    else
    {
        //#pragma omp parallel for
        for(int i=0; i<16; i++)
        {            
            result.sign_array[i]=weight[Cout_index*conv_filter_size+(16*z+i)*kernel_size*kernel_size + x*kernel_size + y].sign;
            int diff =  temp_max_exp- weight[Cout_index*conv_filter_size+(16*z+i)*kernel_size*kernel_size + x*kernel_size + y].exp;
            result.mant_array[i] = weight[Cout_index*conv_filter_size+(16*z+i)*kernel_size*kernel_size + x*kernel_size + y].mant >> diff;            

        }

    } 
    return result;
}

struct BFP form_BFP_direct_i(struct bfp_data * input, int Cin1, int j, int Bout_index, int Cin, int Hout_index, int Wout_index, int stride, int kernel_size, int a_in_padded, int b_in_padded, int c_in_padded, int isfirst, int x, int y, int z)
{
    struct BFP result;
    int temp_max_exp= -1000;

    int barrier= Cin1*kernel_size*kernel_size;
    int b_conv= kernel_size*kernel_size;
   
    if (isfirst==1)
    {
        for(int i=0; i<16; i++)
        { 
            int jj = j*16+i;
            if(jj<barrier)
            {
                if(temp_max_exp<input[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)].exp)
                {
                    temp_max_exp= input[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)].exp;
                }
            }
            
        }
    }
    else
    {
        for(int i=0; i<16; i++)
        { 
            if(temp_max_exp<input[Bout_index*a_in_padded + ((16*z+i))*b_in_padded + (Hout_index*stride + x)*c_in_padded + (Wout_index*stride + y)].exp)
            {
                temp_max_exp= input[Bout_index*a_in_padded + ((16*z+i))*b_in_padded + (Hout_index*stride + x)*c_in_padded + (Wout_index*stride + y)].exp;
            }
        }

    }
    
    result.g=16;
    result.max_exp=temp_max_exp;

    if(isfirst==1)
    {
        //#pragma omp parallel for
        for(int i=0; i<16; i++)
        {
            int jj= j*16+i;
            if(jj<barrier)
            {
                result.sign_array[i]= input[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)].sign;
                int diff =  temp_max_exp - input[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)].exp;
                result.mant_array[i]= input[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)].mant >> diff;
            }
            else
            {
                result.sign_array[i]=0;
                result.mant_array[i]=0;
            }        

        } 

    } 
    else
    {
        //#pragma omp parallel for
        for(int i=0; i<16; i++)
        {       
            result.sign_array[i]= input[Bout_index*a_in_padded + ((16*z+i))*b_in_padded + (Hout_index*stride + x)*c_in_padded + (Wout_index*stride + y)].sign;
            int diff =  temp_max_exp - input[Bout_index*a_in_padded + ((16*z+i))*b_in_padded + (Hout_index*stride + x)*c_in_padded + (Wout_index*stride +y)].exp;
            result.mant_array[i]= input[Bout_index*a_in_padded + ((16*z+i))*b_in_padded + (Hout_index*stride + x)*c_in_padded + (Wout_index*stride + y)].mant >> diff;
        
        } 

    }
    
    return result;

}


struct BFPImage * BFPConv(struct BFPImage* IN, int Cin, int Cout, int kernel_size, int stride, int padding,  cJSON* cjsonp, cJSON* cjsonp2, const char* name, int mant_bit_size, int group_size)
{
    int B = IN->B;
    int C = IN->C;
    int H = IN->H;
    int W = IN->W;
    struct BFPImage* OUT= (struct BFPImage *)malloc(sizeof(struct BFPImage));
    OUT->B = IN->B;
    OUT->C = Cout;
    OUT->H = ((IN->H)+2*padding-kernel_size)/stride + 1;
    OUT->W = ((IN->W)+2*padding-kernel_size)/stride + 1;

    int Hout = OUT->H;
    int Wout = OUT->W;

    OUT->data = (struct bfp_data*)malloc(sizeof(struct bfp_data)*B*Cout*Hout*Wout);
    
    char * weight_name= (char*)malloc(sizeof(char)*(50));
    char * bias_name = (char*)malloc(sizeof(char)*(50));

    int weight_size = Cout*Cin*kernel_size*kernel_size;
    int bias_size= Cout;
    int bias_exist = 0;
    struct bfp_data * weight = (struct bfp_data*)malloc(sizeof(struct bfp_data)*weight_size);
    struct bfp_data * bias =  (struct bfp_data*)malloc(sizeof(struct bfp_data)*bias_size);

    //////WEIGHT INITIALIZATION
    int b= kernel_size*kernel_size;
    int a= b*Cin;
    int c= kernel_size;
    
    int isfirst= (name=="conv1")?1:0;
   
    strcpy(weight_name, name);
    strcat(weight_name, ".weight");

    //cJSON *item = cJSON_GetObjectItemCaseSensitive(cjsonp2, weight_name);
    //mant_bit_size = ((item)->valueint)-8-1;

    cJSON * layer_weight = cJSON_GetObjectItem(cjsonp, weight_name);
    if((cJSON_GetArraySize(layer_weight)!=Cout)||(cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0))!=Cin)||(cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0))!=kernel_size)||(cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0), 0))!=kernel_size))
    {
        printf("Current Weight Matrix size is  %d x %d x %d x %d , But Real Weight Matrix size is %d x %d x %d x %d !\n", cJSON_GetArraySize(layer_weight), cJSON_GetArraySize(cJSON_GetArrayItem(layer_weight, 0)),cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0)), cJSON_GetArraySize(cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, 0), 0), 0)), Cout, Cin, kernel_size, kernel_size);
        printf("Size does not match!\n");
        return NULL;
    }
    
    #pragma omp parallel for
    for(int i=0; i<weight_size; i++)
    {
        cJSON* value = cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(cJSON_GetArrayItem(layer_weight, i/a), (i/b)%Cin), (i/c)%c), i%c);
        weight[i] = float2bfp((float)(value->valuedouble), mant_bit_size);
    }

    //////BIAS INITIALIZATION
    strcpy(bias_name, name);
    strcat(bias_name, ".bias");
    cJSON * layer_bias = cJSON_GetObjectItem(cjsonp, bias_name);
    if(layer_bias != NULL)
    {
        bias_exist =1;
        if(cJSON_GetArraySize(layer_bias)!=Cout)
        {
            printf("Current Bias size is  %d , But Real Bias size is %d!\n", cJSON_GetArraySize(layer_bias), Cout);
            printf("Size does not match!\n");
            return NULL;
        }
        #pragma omp parallel for
        for(int i=0; i<bias_size; i++)
        {
            cJSON* value = cJSON_GetArrayItem(layer_bias, i);
            bias[i] = float2bfp((float)(value->valuedouble), mant_bit_size);
        }

    }
   // printf("Process1!\n");
    // PADDING OPERATION
    int PADDED_H = H+2*padding;
    int PADDED_W = W+2*padding;    
    int a_in= Cin*H*W;
    int b_in= H*W;
    int c_in= W;
    int a_in_padded= Cin*PADDED_H*PADDED_W;
    int b_in_padded= PADDED_H*PADDED_W;
    int c_in_padded= PADDED_W;

    struct BFPImage* PADDED_IN = (struct BFPImage*)malloc(sizeof(struct BFPImage));    
    
    int IN_size  = B*Cin*H*W;
    int PADDED_IN_size = B*Cin*PADDED_H*PADDED_W;
    PADDED_IN->data = (struct bfp_data*)malloc(sizeof(struct bfp_data)*B*Cin*PADDED_H*PADDED_W);
    
    //memset(PADDED_IN->data, 0.0, sizeof(float)*B*Cin*PADDED_H*PADDED_W);
    #pragma omp parallel for 
    for(int i=0; i<PADDED_IN_size;i++)
    {
        PADDED_IN->data[i].sign = false;
        PADDED_IN->data[i].exp = 0;
        PADDED_IN->data[i].mant = 0;
    }

    #pragma omp parallel for 
    for(int i=0; i< IN_size; i++)
    {
        PADDED_IN->data[a_in_padded*(i/a_in) + b_in_padded*((i/b_in)%Cin) + c_in_padded*((i/c_in)%c_in + padding) + i%c_in+padding] = IN->data[i];
    }
    
    free(IN->data);
    free(IN);
    //printf("BFProcess2!\n");
    ////CONVOLUTION
    int a_out= Cout*Hout*Wout;
    int b_out= Hout*Wout;
    int c_out= Wout;


    int out_size = B*a_out;
    int conv_filter_size = Cin*kernel_size*kernel_size;
    int num_tile = (Cin==3)?10:(conv_filter_size/group_size);
   
    
    int b_conv = kernel_size*kernel_size;
    int aa= num_tile/(kernel_size*kernel_size);

    //struct bfp_data* weight_chunk = (struct bfp_data*)malloc(sizeof(struct bfp_data)*conv_filter_size);
    //struct bfp_data* input_chunk = (struct bfp_data*)malloc(sizeof(struct bfp_data)*conv_filter_size);
    //struct bfp_data* temp_w= (struct bfp_data*)malloc(sizeof(struct bfp_data)*group_size);
    //struct bfp_data* temp_i=(struct bfp_data*)malloc(sizeof(struct bfp_data)*group_size);
    
    //struct bfp_data weight_chunk[conv_filter_size];
    //struct bfp_data input_chunk[conv_filter_size];

    #pragma omp parallel for
    for(int i=0; i<out_size; i++)
    {
       // struct bfp_data temp_conv = {0};
       float temp_conv=0.0;
	    int Bout_index = i/a_out;
        int Cout_index = (i/b_out)%Cout;
        int Hout_index = (i/c_out)%c_out;
        int Wout_index = i%c_out;
        //printf("BFProcess2!\n");
        //struct bfp_data* weight_chunk = (struct bfp_data*)malloc(sizeof(struct bfp_data)*conv_filter_size);
        //struct bfp_data* input_chunk = (struct bfp_data*)malloc(sizeof(struct bfp_data)*conv_filter_size);
        //printf("BFProcess11!\n");
       // struct bfp_data weight_chunk[conv_filter_size];
        //struct bfp_data input_chunk[conv_filter_size];

        //#pragma omp parallel for
        //for(int j=0; j< conv_filter_size; j++)
       // {
            //weight_chunk[j]= weight[Cout_index*a+j];
            //input_chunk[j]= PADDED_IN->data[Bout_index*a_in_padded + ((j/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (j/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + j%kernel_size)];
       // }
        //printf("BFProcess3%d!\n", i);
       
        int intermediate_sum_bit_size =20; 
        //printf("Num tile is %d\n", num_tile);
        #pragma omp parallel for firstprivate( Bout_index, Cout_index, Hout_index, Wout_index)
        for (int j = 0; j < num_tile; j++)
        {
           // printf("BFProcess7!\n");
            //printf("the value of j is %d\n",j);
            int x,y,z;
           // printf("the value of x is %d\n",x);
            if(Cin!=3)
            {
                x= j/(aa*kernel_size);
                y= (j/aa)%kernel_size;
                z= j% aa;
            }
            // printf("BFProcess7!\n");
            //struct bfp_data* temp_w= (struct bfp_data*)malloc(sizeof(struct bfp_data)*group_size);
           //struct bfp_data* temp_i=(struct bfp_data*)malloc(sizeof(struct bfp_data)*group_size);
            //struct bfp_data temp_w[16];
           // struct bfp_data temp_i[16];
           // printf("BFProcess2!\n");
            //form_group(x,y,z, weight_chunk, group_size, kernel_size, &temp_w);
            //if(Cin==3){form_group_basic(j,weight_chunk, group_size, kernel_size, (temp_w));}
            //else {form_group(x,y,z, weight_chunk, group_size, kernel_size, temp_w);}
            //printf("BFProcess3!\n");
            //form_group(x,y,z, input_chunk, group_size, kernel_size, &temp_i);
           // if(Cin==3) {form_group_basic(j,input_chunk, group_size, kernel_size, (temp_i));}
           // else{form_group(x,y,z, input_chunk, group_size, kernel_size, temp_i);}
            //printf("BFProcess4!\n");
            //struct BFP BFP_w = form_BFP(temp_w, group_size);
            struct BFP BFP_w = form_BFP_direct_w(weight, Cin, Cout_index,conv_filter_size, j, isfirst, kernel_size, x, y, z);
            struct BFP BFP_i = form_BFP_direct_i(PADDED_IN->data, Cin, j, Bout_index, Cin, Hout_index, Wout_index, stride, kernel_size, a_in_padded, b_in_padded, c_in_padded, isfirst, x, y, z);
	    
	    struct bfp_data result = bfpmul(BFP_w, BFP_i, intermediate_sum_bit_size);
            
	    //printf("BFProcess5!\n");
            //struct BFP BFP_i = form_BFP(temp_i,group_size);
            //temp_conv+= weight[Cout_index*a+j]*(PADDED_IN->data[Bout_index*a_in_padded + ((j/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (j/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + j%kernel_size)]);
            /* 
            int barrier = 3*7*7;
            float a2a=0.0;
	    for(int q=0; q<16; q++)
            {
              int jj= 16*j +q;		    
	      if(jj<barrier) a2a+= bfp2float(weight[Cout_index*conv_filter_size+16*j+q], mant_bit_size)* bfp2float(PADDED_IN->data[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)], mant_bit_size);
	    }		    
	    
	    struct bfp_data result = bfpmul(BFP_w, BFP_i, intermediate_sum_bit_size);
            
	    bool b1= (a2a>=0)?false:true;
	    bool b2= (bfp2float_aftermul(result, mant_bit_size)>=0)?false:true;
            
	    if(b1!=b2)
	    {
	       	    
	      if(name == "conv1")  printf("A1 is %f\n",a2a);
	      if(name == "conv1")  printf("A2 is %f\n",bfp2float_aftermul(result, mant_bit_size));
	       printf("BFP_W!//////\n");
	       printf("exp is %d\n",BFP_w.max_exp);
	       for(int q=0; q<16;q++)
               {
		    
                  printf("%d ", BFP_w.mant_array[q]);
               }
	       printf("\n");
               for(int q=0; q<16;q++)
               {

                  printf("%d ", BFP_w.max_exp- weight[Cout_index*conv_filter_size+16*j+q].exp);
               }
               printf("\n");
               printf("MANT\n");
	        for(int q=0; q<16;q++)
               {

                  printf("%d ", weight[Cout_index*conv_filter_size+16*j+q].mant);
               }
               printf("\n");
               printf("JJJJJJJ is %d!!!!\n", j);
               printf("SIGN\n");
	       for(int q=0; q<16;q++)
               {

                  printf("%d ", BFP_w.sign_array[q]);
               }

	       printf("\n");
               printf("BFP_I!//////\n");
               for(int q=0; q<16;q++)
               {
                  printf("%f ", bfp2float(weight[Cout_index*conv_filter_size+16*j+q], mant_bit_size));
               }

	       printf("WEIGHT!//////\n");	    
	       for(int q=0; q<16;q++)
	       {
		  printf("%f ", bfp2float(weight[Cout_index*conv_filter_size+16*j+q], mant_bit_size));     
	       }
               printf("\n");	       
	     printf("INPUT!//////\n");
               for(int q=0; q<16;q++)
               {
		  int jj= 16*j+q;
                  printf("%f ",  bfp2float(PADDED_IN->data[Bout_index*a_in_padded + ((jj/b_conv)%Cin)*b_in_padded + (Hout_index*stride + (jj/kernel_size)%kernel_size)*c_in_padded + (Wout_index*stride + jj%kernel_size)], mant_bit_size));
               }
               printf("\n");

	    }*/

	   // if(name == "conv1")  printf("A1 is %f\n",a2a);
	   // if(name == "conv1")  printf("A2 is %f\n",bfp2float_aftermul(result, mant_bit_size));
	    
	    
	   // if(name == "conv1")  printf("A1 is %f\n",bfp2float_aftermul(temp_conv, mant_bit_size)+bfp2float_aftermul(result, mant_bit_size));
            //temp_conv= bfpadd(temp_conv, result, mant_bit_size);
            temp_conv+=bfp2float_aftermul(result, mant_bit_size);
	    // if (name=="conv1")  printf("A2 is %f\n",bfp2float_aftermul(temp_conv, mant_bit_size));
            
	     //printf("BFProcess6!\n");
            //free(temp_w);
            //free(temp_i);
            //free(BFP_w.mant_array);
            //free(BFP_i.mant_array);
           //free(BFP_w.sign_array);
            //free(BFP_i.sign_array);
        }
       // float a1 = bfp2float_aftermul(temp_conv, mant_bit_size);
      //  printf("A1 is %f\n",a1);
       // temp_conv= float2bfp(a1, mant_bit_size);
       // printf("A2 is %f\n",bfp2float(temp_conv, mant_bit_size));
        if(bias_exist==1)
        {
            //temp_conv= bfpadd(temp_conv,bias[Cout_index], mant_bit_size);
	    temp_conv+= bfp2float(bias[Cout_index], mant_bit_size);
        }
        OUT->data[i] = float2bfp(temp_conv, mant_bit_size);

        //free(weight_chunk);
        //free(input_chunk); 
        
    }

   // free(temp_w);
   // free(temp_i);
   // free(BFP_w.mant_array);
   // free(BFP_i.mant_array);
   // free(BFP_w.sign_array);
    //free(BFP_i.sign_array);
    //free(weight_chunk);
    //free(input_chunk); 


    
    free(weight);
    free(weight_name);
    free(bias);
    free(bias_name);
    free(PADDED_IN->data);
    free(PADDED_IN);  
    
    return OUT;    
} 


struct Image* BFPBlock(struct Image* IN, int Cin, int Cout, int downsample,int mant_bit_size, cJSON* cjsonp, cJSON* cjsonp2, const char* name)
{    
    char  conv0_name[50];
    char  bn0_name[50];
    char  conv1_name[50];
    char  bn1_name[50];
    char  ds_conv0_name[50];
    char  ds_bn0_name[50];
    float eps = 0.00001;

    strcpy(conv0_name, name);
    strcpy(bn0_name, name);
    strcpy(conv1_name, name);
    strcpy(bn1_name, name);
    strcpy(ds_conv0_name, name);
    strcpy(ds_bn0_name, name);


    strcat(conv0_name, ".conv1");
    strcat(bn0_name, ".bn1");
    strcat(conv1_name, ".conv2");
    strcat(bn1_name, ".bn2");
    strcat(ds_conv0_name, ".downsample.0");
    strcat(ds_bn0_name, ".downsample.1");


    struct Image * IN1 = (struct Image*)malloc(sizeof(struct Image));
    struct BFPImage* BFPIN;
    int IN_size = (IN->B)*(IN->C)*(IN->H)*(IN->W);
    IN1->data= (float*)malloc(sizeof(float)*IN_size);
    IN1->B = IN->B;
    IN1->C = IN->C;
    IN1->H = IN->H;
    IN1->W = IN->W;

    #pragma omp parallel for
    for(int i=0; i< IN_size; i++)
    {
        IN1->data[i] = IN->data[i];
    }    
   // printf("AA0\n");

    cJSON *item = cJSON_GetObjectItemCaseSensitive(cjsonp2, conv0_name);
    mant_bit_size = ((item)->valueint)-8-1;

    
    BFPIN = Image2BFPImage(IN,mant_bit_size);
    BFPIN = BFPConv(BFPIN, Cin, Cout, 3, (downsample==1)?2:1,1, cjsonp, cjsonp2, conv0_name, mant_bit_size, 16);    
   // printf("AA1\n");
    IN= BFPImage2Image(BFPIN, mant_bit_size);
    
    IN = Batchnorm(IN, cjsonp, bn0_name, eps);    
    //  printf("AA2\n");
    IN = ReLU(IN);
   // printf("AA3\n");

    item = cJSON_GetObjectItemCaseSensitive(cjsonp2, conv1_name);
    mant_bit_size = ((item)->valueint)-8-1;

    BFPIN = Image2BFPImage(IN,mant_bit_size);
    BFPIN = BFPConv(BFPIN, Cout, Cout, 3, 1,1, cjsonp, cjsonp2, conv1_name, mant_bit_size, 16);
    IN= BFPImage2Image(BFPIN, mant_bit_size);
   
  // IN = Conv(IN, Cout, Cout, 3, 1, 1, cjsonp, conv1_name);

   // printf("AA4\n");
    IN = Batchnorm(IN, cjsonp, bn1_name, eps); 
    //printf("AA5\n");

    item = cJSON_GetObjectItemCaseSensitive(cjsonp2, ds_conv0_name);
    mant_bit_size = ((item)->valueint)-8-1;

    return ReLU((downsample==0)?ADD(IN, IN1):ADD(IN,Batchnorm(BFPImage2Image(BFPConv( Image2BFPImage(IN1,mant_bit_size), Cin ,Cout, 1, 2, 0, cjsonp, cjsonp2, ds_conv0_name, mant_bit_size, 16), mant_bit_size),cjsonp, ds_bn0_name, eps)));
    
    //return IN;    
    }

struct Image* ForwardBFPResnet18(struct Image* IN, cJSON* root, cJSON* root2, int mant_bit_size)
{
    float eps =0.00001;
    struct BFPImage * BFP_IN;
    
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root2, "conv1");
    mant_bit_size = ((item)->valueint)-8-1;

    BFP_IN= Image2BFPImage(IN, mant_bit_size);
    //printf("P0\n");
    BFP_IN= BFPConv(BFP_IN, 3, 64, 7, 2, 3, root, root2, "conv1", mant_bit_size, 16);
    //  printf("P1\n");
   
    IN= BFPImage2Image(BFP_IN, mant_bit_size);
   
    IN = Batchnorm(IN, root,"bn1", eps);
    IN = ReLU(IN);
    IN = MaxPool(IN, 3, 2, 1);
  //  printf("P2\n");
    IN = BFPBlock(IN, 64, 64 , 0, mant_bit_size, root, root2, "layer1.0");
   //  IN = Block(IN, 64, 64 , 0, root, "layer1.0");
  //  printf("P3\n");
    IN = BFPBlock(IN, 64, 64 , 0, mant_bit_size, root, root2, "layer1.1");
  //  IN = Block(IN, 64, 64 , 0, root, "layer1.1");
  //  printf("P4\n");
    IN = BFPBlock(IN, 64, 128 , 1, mant_bit_size, root, root2, "layer2.0");
  //  IN = Block(IN, 64, 128 , 1, root, "layer2.0");
    //printf("P5\n");
   IN = BFPBlock(IN, 128, 128 , 0, mant_bit_size, root, root2, "layer2.1");
  //  IN = Block(IN, 128, 128, 0, root, "layer2.1");
   // printf("P6\n");
    IN = BFPBlock(IN, 128, 256 , 1, mant_bit_size, root, root2, "layer3.0");
   // IN = Block(IN, 128, 256, 1, root, "layer3.0");
    IN = BFPBlock(IN, 256, 256 , 0, mant_bit_size, root, root2, "layer3.1");
   // IN = Block(IN, 256, 256, 0, root, "layer3.1");
    //printf("P7\n");
    IN = BFPBlock(IN, 256, 512 , 1, mant_bit_size, root, root2, "layer4.0");
   // IN = Block(IN, 256, 512, 1, root, "layer4.0");
    IN = BFPBlock(IN, 512, 512 , 0, mant_bit_size, root, root2, "layer4.1");
   // IN = Block(IN, 512, 512, 0, root, "layer4.1");
    
    IN = AveragePool(IN, 7, 0);
   //IN = Linear(IN, 512, 100, root, "fc");
    //printf("P8\n");
    
    item = cJSON_GetObjectItemCaseSensitive(root2, "fc");
    mant_bit_size = ((item)->valueint)-8-1;


    BFP_IN= Image2BFPImage(IN, mant_bit_size);
    //printf("P9\n");
    BFP_IN = BFPLinear(BFP_IN, 512, 100, mant_bit_size, 16, root, root2, "fc");
    //printf("P10\n");
    IN= BFPImage2Image(BFP_IN, mant_bit_size);
    IN = ArgMax(IN);
    //printf("P11\n");
    
    return IN;
}



void Initdataloader( struct Dataloader* dataloader, int batch_size, int total_image, int shuffle)
{
    //printf("A\n");
    //Transform(&dataset, 224, total_image);
    //printf("B\n");
   
    dataloader->total_image = total_image;    
    dataloader->batch_size = batch_size;
    dataloader->current_index = 0;
    dataloader->shuffle = shuffle;

    // Initialize indices for shuffling
    dataloader->indices = (int *)malloc(total_image * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < total_image; i++) {
        dataloader->indices[i] = i;
    }

    if (shuffle) {
        // Shuffle the indices
        srand(time(NULL));
        #pragma omp parallel for
        for (int i = total_image - 1; i > 0; i--) 
        {
            int j = rand() % (i + 1);
            int temp = dataloader->indices[i];
            dataloader->indices[i] = dataloader->indices[j];
            dataloader->indices[j] = temp;
        }
    }
    return;

}

int get_next_batch(struct Dataloader *dataloader) 
{
    if (dataloader->current_index >= dataloader->total_image)
    {
        return 0; 
    }

    int start_index = dataloader->current_index;
    int end_index = start_index + dataloader->batch_size;
    if (end_index > dataloader->total_image) 
    {
        end_index = dataloader->total_image;
    }

    int current_batch_size = end_index - start_index;
    //printf("A1\n");
    Transform(&(dataloader->resized_images), dataloader->raw_image, current_batch_size, 224, dataloader->current_index);   
    //printf("A2\n");
    dataloader->current_index = end_index;
    return current_batch_size;
}


int main()
{   
    omp_set_num_threads(40);
    srand(time(NULL));
    //omp_set_nested(0);
    float eps = 0.00001;
    const char * filename = "weights_and_biases_v8.json";
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);    
    char *json_str = (char *)malloc(fsize + 1);
    fread(json_str, 1, fsize, fp);
    fclose(fp);
    json_str[fsize] = '\0';

    /////////Eiglist

    const char * filename2 = "eiglist_v2.json";
    FILE *fp2 = fopen(filename2, "r");
    if (!fp2) {
        printf("Error opening file\n");
        return 1;
    }
    fseek(fp2, 0, SEEK_END);
    long fsize2 = ftell(fp2);
    fseek(fp2, 0, SEEK_SET);    
    char *json_str2 = (char *)malloc(fsize2 + 1);
    fread(json_str2, 1, fsize2, fp2);
    fclose(fp2);
    json_str2[fsize2] = '\0';
    cJSON *root2 = cJSON_Parse(json_str2);
    if (!root2) {
        printf("Error parsing JSON\n");
        return 0;        
    }
    


    //printf("%ld",fsize);
   
    //printf("Error!!\n");
    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("Error parsing JSON\n");
        return 0;        
    }
    struct Image *IN = (struct Image*)malloc(sizeof(struct Image));
    IN->B = 20;
    IN->H = 224;
    IN->W = 224;
    IN->C = 3;

    struct Image *IN_D = (struct Image*)malloc(sizeof(struct Image));
    IN_D->B = 20;
    IN_D->H = 224;
    IN_D->W = 224;
    IN_D->C = 3;
   
    int whole_size = (IN->B)*(IN->C)*(IN->H)*(IN->W);

   // struct Image *IN2 = (struct Image*)malloc(sizeof(struct Image));
    struct Image *IN3 = (struct Image*)malloc(sizeof(struct Image));

    //struct Image *IN1 = (struct Image*)malloc(sizeof(struct Image));
    

    IN->data = (float*)malloc(sizeof(float)*whole_size);
    IN_D->data = (float*)malloc(sizeof(float)*whole_size);
    //IN1->data = (float*)malloc(sizeof(float)*(2*56*56*64));
    //IN2->data = (float*)malloc(sizeof(float)*(2*56*56*64));

    for(int i = 0;  i< whole_size;i++)
    {
        IN->data[i] =77;
        IN_D->data[i]= 77;
      
    }
    long int position = 0;
    clock_t t;
    t = clock();
    int msfp_format=16;
    int exp_bit_size=8;
    int group_size =16;
    int mant_bit_size= msfp_format - exp_bit_size -1;
    struct Image* IN1;
    struct Image* IN2;
    //IN = Block(IN, 64, 128 , 1, root, "layer2.0");    
    //IN = ForwardResnet18(IN,root);
    //struct Limage* test_dataset;
    struct Dataloader* testloader = (struct Dataloader*) malloc(sizeof(struct Dataloader));  
    //Readdataset("cifar100/test.bin", &test_dataset, 10000);
    Initdataloader(testloader, 64, 10000, 0);
    int num=0;
    int num2=0;
    Readbatch("cifar100/test.bin", testloader, &position);
    int fetched = 0;
    while((fetched = get_next_batch(testloader)) > 0)
    {
       
        //printf("first flabel is %u\n", testloader->flabel[0]);
        //num++;
        printf("Batch Processed: %d\n", num);
       
        if(testloader->flabel[0]>=100)
        {
            printf("LABEL ERROR!\n");
            return 0;
        }
        /*
	struct Image* test1 = (struct Image*)malloc(sizeof(struct Image));
        int whole_s= 64*3*224*224;
        test1->data= (float*)malloc(sizeof(float)* whole_s);
        test1->B= 64;
	test1->C= 3;
        test1->H= 224;
	test1->W= 224;
	for(int i=0; i<whole_s;i++)
	{
           test1->data[i]= testloader->resized_images->data[i];		
	}	
       // printf("BF\n");
          //IN1= ForwardResnet18(test1, root, num);
          //IN2= ForwardBFPResnet18(testloader->resized_images, root, mant_bit_size);
	  testloader->resized_images= ForwardBFPResnet18(testloader->resized_images, root, mant_bit_size);
       // printf("tested flabel  is %u and real flabel is  %u\n",(unsigned char)testloader->resized_images->data[0],  testloader->flabel[0]);
        
	for(int i=0; i<100; i++)
        {
            if(i%8==0)
            {
                printf("\n");
            }
            printf("%.10f ", IN1->data[i]);
            
        }
        printf("\n");
	printf("///////////////////////////////////////////////\n");

        for(int i=0; i<100; i++)
        {
            if(i%8==0)
            {
                printf("\n");
            }
            printf("%.10f ", IN2->data[i]);

        }
        */ 
        
	testloader->resized_images= ForwardBFPResnet18(testloader->resized_images, root, root2, mant_bit_size);
        
	for(int i=0; i<fetched;i++)
        {
           if(testloader->flabel[i]==(unsigned char)(testloader->resized_images->data[i]))
           {
              /*if((unsigned char)(IN1->data[i]) != (unsigned char)(IN2->data[i]))
              {
		printf("the real label is %u but the bfp model's label is %u!\n",(unsigned char)(IN1->data[i]),(unsigned char)(IN2->data[i]));      
		printf("Batch is %d and the index is %d\n", num,i);
		printf("\n\n");
	      
	      }	*/	      
              num2++;
           }
	}

        num++;
        //printf("AF\n");
	
        free(testloader->flabel);
        free(testloader->clabel);
        free(testloader->raw_image);
        free(testloader->resized_images->data);
        free(testloader->resized_images);
        testloader->resized_images = NULL;
        
	/*num++;
	if(num==1)
        {
            break;
        }*/
        Readbatch("cifar100/test.bin", testloader, &position);
    }
    
    printf("The number of matched image is  %d\n", num2);
    printf("Test Accuracy is %f%%\n", 100*(((float)num2)/(10000)));
    
    t = clock() - t;
    
    printf("%d\n", IN->B);
    printf("%d\n", IN->C);
    printf("%d\n", IN->H);
    printf("%d\n", IN->W);
    whole_size = (IN->B)*(IN->C)*(IN->H)*(IN->W);
    for(int i = 0;  i< whole_size;i++)
    {
       // printf("%f ", IN->data[i]);        
    }
    printf("\n");
    

    double time_taken = ((double)t)/CLOCKS_PER_SEC; // calculate the elapsed time
    printf("The program took %f seconds to execute", time_taken);
    /*
    int num7=0;
    #pragma omp parallel for
    for(int i=0; i<64; i++)
    {
        num7++;
    }
    printf("num7 value is %d\n", num7);
    */
    return 0;
}

