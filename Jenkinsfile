#!groovy

node {

    load "$JENKINS_HOME/jobvars.env"

    dir('src/github.com/reportportal/service-auto-analyzer') {

        stage('Checkout') {
            checkout scm
        }

        stage('Build') {
            withEnv(["IMAGE_POSTFIX=-dev", "BUILD_NUMBER=${env.BUILD_NUMBER}", "DOCKER_BUILDKIT=1"]) {
                docker.withServer("$DOCKER_HOST") {
                    stage('Test Docker Image') {
                        sh """
                            make build-image-test
                            make run-test
                        """
                    }
                    stage('Build Docker Image') {
                        sh """
                            MAJOR_VER=\$(cat VERSION)
                            BUILD_VER="\${MAJOR_VER}-${env.BUILD_NUMBER}"
                            make build-image-dev v=\$BUILD_VER
                        """
                    }
                }
            }
        }

    }
}