pipeline {
    agent any

   
    

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'make install'
                sh 'make install-sonar'
            }
        }
     

        stage('Start MLflow Server') {
            steps {
                sh 'venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &'
                sh 'sleep 10' // Wait for the server to start
            }
        }

        stage('Debug Dependencies') {
            steps {
                sh 'venv/bin/pip list | grep urllib3'
                sh 'venv/bin/pip show urllib3'
            }
        }
        
        stage('SonarQube Analysis') {
    steps {
        script {
            
            sh 'export PATH=$PATH:/opt/sonar-scanner/bin && make sonar'
        }
    }
}

        stage('Run Tests') {
            steps {
                sh '''
                    make unit-test
                    make functional-test
                '''
            }
        }

        

        stage('Data Pipeline') {
            steps {
                sh 'make data'
            }
        }

        stage('Training Pipeline') {
            steps {
                sh 'make train'
            }
        }

        stage('Evaluation Pipeline') {
            steps {
                sh 'make evaluate'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'make docker-build'
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_HUB_USER', passwordVariable: 'DOCKER_HUB_PASSWORD')]) {
                    sh 'docker login -u $DOCKER_HUB_USER -p $DOCKER_HUB_PASSWORD'
                    sh 'make docker-push'
                }
            }
        }

        stage('Deploy') {
            steps {
                sh 'make docker-run'
            }
        }
    }

    post {
        always {
            // Clean up to save disk space
            sh 'docker system prune -f || true'
            sh 'find . -name "__pycache__" -type d -exec rm -rf {} +  || true'
            sh 'find . -name "*.pyc" -delete || true'
        }
        success {
            emailext (
                body: """
                <html>
                <body>
                <h2>✅ Pipeline Successful</h2>
                <p>Build: ${env.BUILD_NUMBER}</p>
                <p>Check console output at <a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a></p>
                </body>
                </html>
                """,
                subject: "✅ Pipeline Success: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                to: 'bennourines00@gmail.com',
                mimeType: 'text/html'
            )
        }
        failure {
            emailext (
                body: """
                <html>
                <body>
                <h2>❌ Pipeline Failed</h2>
                <p>Build: ${env.BUILD_NUMBER}</p>
                <p>Check console output at <a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a></p>
                </body>
                </html>
                """,
                subject: "❌ Pipeline Failed: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                to: 'bennourines00@gmail.com',
                mimeType: 'text/html'
            )
        }
    }
}
