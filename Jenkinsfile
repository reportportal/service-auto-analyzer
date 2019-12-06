#!groovy

node {

    load "$JENKINS_HOME/jobvars.env"

    dir('src/github.com/reportportal/service-auto-analyzer') {

        stage('Checkout') {
            checkout scm
        }

        stage('Build') {
            withEnv(["IMAGE_POSTFIX=-dev", "BUILD_NUMBER=${env.BUILD_NUMBER}"]) {
                docker.withServer("$DOCKER_HOST") {
                    stage('Build Docker Image') {
                        sh """
                            MAJOR_VER=\$(cat VERSION)
                            BUILD_VER="\${MAJOR_VER}-${env.BUILD_NUMBER}"
                            make build build-image-dev v=\$BUILD_VER
                        """
                    }
                }
            }
        }

    }
}