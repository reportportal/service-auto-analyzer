#!groovy

node {

    load "$JENKINS_HOME/jobvars.env"

    stage('Checkout') {
        checkout scm
    }

    stage('Build Docker Image') {
        sh """
        MAJOR_VER=\$(cat VERSION)
        BUILD_VER="\${MAJOR_VER}-${env.BUILD_NUMBER}"
        make build-image-dev v=\$BUILD_VER
        """
    }
    stage('Deploy Container') {
        stage('Push to ECR') {
            withEnv(["AWS_URI=${AWS_URI}", "AWS_REGION=${AWS_REGION}"]) {
                sh 'docker tag reportportal-dev/service-auto-analyzer ${AWS_URI}/service-auto-analyzer:SNAPSHOT-${BUILD_NUMBER}'
                def image = env.AWS_URI + '/service-auto-analyzer'+ ':SNAPSHOT-' + env.BUILD_NUMBER
                def url = 'https://' + env.AWS_URI
                def credentials = 'ecr:' + env.AWS_REGION + ':aws_credentials'
                docker.withRegistry(url, credentials) {
                    docker.image(image).push()
                }
            }
        }
        
        stage('Cleanup') {
            withEnv(["AWS_URI=${AWS_URI}"]) {
                sh 'docker rmi ${AWS_URI}/service-auto-analyzer:SNAPSHOT-${BUILD_NUMBER}'
                sh 'docker rmi reportportal-dev/service-auto-analyzer:latest'
            }
        }
    }
}
