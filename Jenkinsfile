pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                // Checkout your repository
                script {
                    def checkoutCmd = """git clone ${env.REPOSITORY_URL}"""
                    sh checkoutCmd
                }
            }
        }
        
        stage('Deploy Chatbot App') {
            steps {
                dir('your-repository-directory') {
                    // Replace 'your-repository-directory' with the actual directory containing your app
                    // Install required dependencies and deploy your app
                    sh """
                    sudo apt-get update
                    sudo apt-get install -y python3-pip
                    pip3 install -r requirements.txt
                    python3 train.py
                    python3 app.py
                    """
                }
            }
        }
    }
}
