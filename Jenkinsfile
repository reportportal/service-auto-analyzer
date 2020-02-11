#!groovy

node {

    load "$JENKINS_HOME/jobvars.env"

    stage('Checkout') {
        checkout scm
    }

    docker.withServer("$DOCKER_HOST") {
        stage('Build Docker Image') {
            sh """
                            MAJOR_VER=\$(cat VERSION)
                            BUILD_VER="\${MAJOR_VER}-${env.BUILD_NUMBER}"
                            make build-image-dev v=\$BUILD_VER
                        """
        }
        stage('Deploy Container') {
            sh "docker-compose -f $COMPOSE_FILE_RP -p reportportal up -d --force-recreate analyzer"
        }
    }
}
