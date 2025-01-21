# ML Training for Medical Images

This repository contains the code and configurations for training machine learning models on medical imaging datasets. It is designed to help you set up and execute training workflows using various machine learning models in medical image classification or other related tasks.

## Structure

- **Dockerfile.train**: Dockerfile for creating an environment for training models.
- **argo_workflow_ml_train.yaml**: Argo workflow definition for orchestrating machine learning training jobs.
- **requirements.txt**: Python dependencies required for the training pipeline.
- **train_model.py**: Python script to train the machine learning model.
- **validate_model.py**: Python script to validate the trained model.

## Setup


1. Clone this repository:

   ```bash
   git clone https://github.com/dranicu/ml_training_medical_images.git
   ```

2. Install the required Python packages (for local development or if you need the dependencies):

   ```bash
   pip install -r requirements.txt
   ```

3. Build the Docker container:

   ```bash
   docker build -f Dockerfile.train -t ml_training_medical_images .
   ```

4. Run the training inside the container:

   ```bash
   docker run --rm ml_training_medical_images
   ```

   This command will run the training process inside the container, ensuring all dependencies and environment configurations are correctly handled.

## Argo Workflow

This repository includes an Argo workflow (`argo_workflow_ml_train.yaml`) that can be used for automating the training job in an orchestrated environment, such as Kubernetes. Make sure to have an Argo setup configured to run the workflow.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Include any references or acknowledgements to libraries, tools, or authors here.
```

This version now reflects the change to run the training process within the container instead of running a script directly on the host machine.
