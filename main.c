#include "backprop.h"
#include "layer.h"
#include "neuron.h"


layer *lay = NULL;
int num_layers=4;
int *num_neurons;
float learning_rate=0.15;
float *cost;
float full_cost;
float **input;
float **desired_outputs;
int num_training_ex=4;
int n=1;

int main(void)
{
    int i;

    srand(time(0));


    num_neurons = (int*) malloc(num_layers * sizeof(int));
    memset(num_neurons,0,num_layers *sizeof(int));

    // Get number of neurons per layer
    for(i=0;i<num_layers;i++)
    {
        printf("Enter number of neurons in layer[%d]: ",i+1);
        scanf("%d",&num_neurons[i]);
    }

    printf("\n");

    // Initialize the neural network module
    if(init()!= SUCCESS_INIT)
    {
        printf("Error in Initialization...\n");
        exit(0);
    }


    input = (float**) malloc(num_training_ex * sizeof(float*));
    for(i=0;i<num_training_ex;i++)
    {
        input[i] = (float*)malloc(num_neurons[0] * sizeof(float));
    }

    desired_outputs = (float**) malloc(num_training_ex* sizeof(float*));
    for(i=0;i<num_training_ex;i++)
    {
        desired_outputs[i] = (float*)malloc(num_neurons[num_layers-1] * sizeof(float));
    }

    cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));
    memset(cost,0,num_neurons[num_layers-1]*sizeof(float));

    // Get Training Examples and Output Labels
    get_inputs();

    train_neural_net();
    test_nn();

    if(dinit()!= SUCCESS_DINIT)
    {
        printf("Error in Dinitialization...\n");
    }

    return 0;
}


int init()
{
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS_INIT;
}

//Get Inputs and Desired Outputs 
void  get_inputs()
{
    int i,j;

        for(i=0;i<num_training_ex;i++)
        {
            printf("Enter the Inputs for training example[%d]:",i);

            for(j=0;j<num_neurons[0];j++)
            {
                scanf("%f",&input[i][j]);
            }

            // printf("\n");

            for(j=0;j<num_neurons[num_layers-1];j++)
            {
                printf("Enter the Desired Outputs for training example[%d]: ",i);
                scanf("%f",&desired_outputs[i][j]);
                printf("\n");
            }

        }

}

// Feed inputs to input layer
void feed_input(int i)
{
    int j;

    for(j=0;j<num_neurons[0];j+=j+2)
    {
        lay[0].neu[j].actv = input[i][j];
        lay[0].neu[j+1].actv = input[i][j+1];

        printf("Input: (%.2f,%.2f)      ",lay[0].neu[j].actv,lay[0].neu[j+1].actv);
    }
}

// Create Neural Network Architecture
int create_architecture()
{
    int i=0,j=0;
    lay = (layer*) malloc(num_layers * sizeof(layer));

    for(i=0;i<num_layers;i++)
    {
        lay[i] = create_layer(num_neurons[i]);      
        lay[i].num_neu = num_neurons[i];
       
  
        for(j=0;j<num_neurons[i];j++)
        {
            if(i < (num_layers-1)) 
            {
                lay[i].neu[j] = create_neuron(num_neurons[i+1]);
            }

        }
       
    }

    // Initialize the weights
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

int initialize_weights(void)
{
    int i,j,k;

    if(lay == NULL)
    {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing weights...\n");

    for(i=0;i<num_layers-1;i++)
    {
        
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Initialize Output Weights for each neuron
                lay[i].neu[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, lay[i].neu[j].out_weights[k]);
                lay[i].neu[j].dw[k] = 0.0;
            }

            if(i>0) 
            {
                lay[i].neu[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }   
    printf("\n");
    
    for (j=0; j<num_neurons[num_layers-1]; j++)
    {
        lay[num_layers-1].neu[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS_INIT_WEIGHTS;
}

// Train Neural Network
void train_neural_net(void)
{
    int i;
    int it=0;
    int num_training;

    printf("Enter the number for training(usually 20000):");
    scanf("%d",&num_training);


    // Gradient Descent
    for(it=0;it<num_training;it++)
    {
        
        printf("< Training Time: %d >\n" , it+1);
        
        for(i=0;i<num_training_ex;i++)
        {
            
            feed_input(i);
            forward_prop();
            compute_cost(i);
            back_prop(i);
            update_weights();
        }
        if(it==(num_training)-1){
            printf("< You can start testing! >");
            printf("\n");
        }
    }
}


//  Update the Network weights and bias
void update_weights(void)
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (learning_rate * lay[i].neu[j].dw[k]);
            }
            
            // Update Bias
            lay[i].neu[j].bias = lay[i].neu[j].bias - (learning_rate * lay[i].neu[j].dbias);
        }
    }   
}

//Forward Pass
void forward_prop(void)
{
    int i,j,k;

    for(i=1;i<num_layers;i++)
    {   
        for(j=0;j<num_neurons[i];j++)
        {
            lay[i].neu[j].z = lay[i].neu[j].bias;

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i].neu[j].z  = lay[i].neu[j].z + ((lay[i-1].neu[k].out_weights[j])* (lay[i-1].neu[k].actv));
            }

            // Relu Activation Function for Hidden Layers
            if(i < num_layers-1)
            {
                if((lay[i].neu[j].z) < 0)
                {
                    lay[i].neu[j].actv = 0;
                }

                else
                {
                    lay[i].neu[j].actv = lay[i].neu[j].z;
                }
            }
            
            // Sigmoid Activation function for Output Layer
            else
            {
                lay[i].neu[j].actv = 1/(1+exp(-lay[i].neu[j].z));
                printf("Output: %d\n", (int)round(lay[i].neu[j].actv));
                printf("\n");
            }
        }
    }
}

// Compute Total Cost
// Loss function
void compute_cost(int i)
{
    int j;
    float tmpcost=0;
    float tcost=0;

    for(j=0;j<num_neurons[num_layers-1];j++)
    {
        
        tmpcost = desired_outputs[i][j] - lay[num_layers-1].neu[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
    }   

    full_cost = (full_cost + tcost)/n;
    n++;
}

// Back Propogate Error
// Backward Pass
void back_prop(int p)
{
    int i,j,k;

    // Output Layer
    for(j=0;j<num_neurons[num_layers-1];j++)
    {           
        lay[num_layers-1].neu[j].dz = (lay[num_layers-1].neu[j].actv - desired_outputs[p][j]) * (lay[num_layers-1].neu[j].actv) * (1- lay[num_layers-1].neu[j].actv);

        for(k=0;k<num_neurons[num_layers-2];k++)
        {   
            lay[num_layers-2].neu[k].dw[j] = (lay[num_layers-1].neu[j].dz * lay[num_layers-2].neu[k].actv);
            lay[num_layers-2].neu[k].dactv = lay[num_layers-2].neu[k].out_weights[j] * lay[num_layers-1].neu[j].dz;
        }
            
        lay[num_layers-1].neu[j].dbias = lay[num_layers-1].neu[j].dz;           
    }

    // Hidden Layers
    for(i=num_layers-2;i>0;i--)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            if(lay[i].neu[j].z >= 0)
            {
                lay[i].neu[j].dz = lay[i].neu[j].dactv;
            }
            else
            {
                lay[i].neu[j].dz = 0;
            }

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i-1].neu[k].dw[j] = lay[i].neu[j].dz * lay[i-1].neu[k].actv;    
                
                if(i>1)
                {
                    lay[i-1].neu[k].dactv = lay[i-1].neu[k].out_weights[j] * lay[i].neu[j].dz;
                }
            }

            lay[i].neu[j].dbias = lay[i].neu[j].dz;
        }
    }
}

// Test the trained network
void test_nn(void) 
{
    int i;
    while(1)
    {
     
        printf("Enter input to test: ");

        for(i=0;i<num_neurons[0];i++)
        {
            scanf("%f",&lay[0].neu[i].actv);
        }
        forward_prop();
    }
}


int dinit(void)
{
   
    // Free up all the structures
    free(num_neurons);
    free(input);
    free(desired_outputs);
    free(cost);
    free(lay);

    return SUCCESS_DINIT;
}