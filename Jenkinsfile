pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the Git repository
                checkout scm
            }
        }

        stage('Build') {
            steps {
                // Install Python and create a virtual environment
                sh 'sudo apt-get update'
                sh 'sudo apt-get install python3 python3-venv python3-pip -y'
                sh 'python3 -m venv venv' // Create a virtual environment

                // Activate the virtual environment and store the path in an environment variable
                script {
                    def venvPath = "${WORKSPACE}/venv/bin"
                    env.PATH = "${venvPath}:${env.PATH}"
                }

                // Install Flask and other dependencies (if any)
                sh '/usr/bin/pip install Flask'  // Replace with the correct path to pip

                }
        }

        stage('Deploy') {
            steps {
                // Run the chatbot within the virtual environment in the background using nohup
                sh 'nohup python app.py > chatbot.log 2>&1 &'
            }
        }
    }
}
