#!groovy

node {

    load "$JENKINS_HOME/jobvars.env"

    stage('Checkout') {
        checkout scm
    }

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
        stage('Deploy Container') {
            sh "docker-compose up -f $COMPOSE_FILE_RP_5_1 -p reportportal51 -d --force-recreate analyzer"
        }
    }
}