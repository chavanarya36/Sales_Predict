pipeline {
    agent any

    environment {
        IMAGE_NAME = 'sales_predict_image'
        CONTAINER_NAME = 'sales_predict_container'
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/chavanarya36/Sales_Predict.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build(IMAGE_NAME)
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    docker.image(IMAGE_NAME).inside {
                        sh 'pip install -r requirements.txt'
                        sh 'pip install pytest'
                        sh 'pytest tests/'
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
        }
        success {
            echo 'Tests passed!'
        }
        failure {
            echo 'Tests failed.'
        }
    }
}
