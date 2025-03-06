pipeline {
    agent any

    environment {
        SONAR_SCANNER_HOME = '/opt/sonar-scanner'
        PATH = "${env.PATH}:${SONAR_SCANNER_HOME}/bin"
    }

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'make install'
                // Install SonarScanner globally
                sh '''
                    wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.8.0.2856-linux.zip
                    unzip sonar-scanner-cli-4.8.0.2856-linux.zip
                    sudo mv sonar-scanner-4.8.0.2856-linux /opt/sonar-scanner
                '''
            }
        }

        stage('Start MLflow Server') {
            steps {
                sh 'venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &'
                sh 'sleep 10'
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    source venv/bin/activate
                    pip install pytest-cov
                    python -m pytest --cov=. --cov-report=xml:coverage.xml --junitxml=test-results.xml
                '''
            }
        }

        stage('SonarQube Analysis') {
            environment {
                SONAR_TOKEN = credentials('sonarqube-token')
            }
            steps {
                withSonarQubeEnv('SonarQube') {
                    sh """
                        sonar-scanner \
                        -Dsonar.projectKey=mlops-churn-prediction \
                        -Dsonar.host.url=http://localhost:9000 \
                        -Dsonar.login=${SONAR_TOKEN} \
                        -Dsonar.python.coverage.reportPaths=coverage.xml \
                        -Dsonar.python.xunit.reportPath=test-results.xml \
                        -Dsonar.exclusions=venv/**/*,**/*.pyc,**/__pycache__/**,**/test_*.py
                    """
                }
                timeout(time: 1, unit: 'HOURS') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        // Remaining stages unchanged
        stage('Data Pipeline') { steps { sh 'make data' } }
        stage('Training Pipeline') { steps { sh 'make train' } }
        stage('Evaluation Pipeline') { steps { sh 'make evaluate' } }
        stage('Build Docker Image') { steps { sh 'make docker-build' } }
        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_HUB_USER', passwordVariable: 'DOCKER_HUB_PASSWORD')]) {
                    sh 'docker login -u $DOCKER_HUB_USER -p $DOCKER_HUB_PASSWORD'
                    sh 'make docker-push'
                }
            }
        }
        stage('Deploy') { steps { sh 'make docker-run' } }
    }

    post {
        always {
            sh 'docker system prune -f || true'
            sh 'find . -name "__pycache__" -type d -exec rm -rf {} + || true'
            sh 'find . -name "*.pyc" -delete || true'
        }
        success {
            emailext(
                body: """<html><body>
                    <h2>✅ Pipeline Successful</h2>
                    <p>Build: ${env.BUILD_NUMBER}</p>
                    <p>Details: <a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a></p>
                    </body></html>""",
                subject: "✅ Pipeline Success: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                to: 'bennourines00@gmail.com',
                mimeType: 'text/html'
            )
        }
        failure {
            emailext(
                body: """<html><body>
                    <h2>❌ Pipeline Failed</h2>
                    <p>Build: ${env.BUILD_NUMBER}</p>
                    <p>Details: <a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a></p>
                    </body></html>""",
                subject: "❌ Pipeline Failed: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                to: 'bennourines00@gmail.com',
                mimeType: 'text/html'
            )
        }
    }
}
