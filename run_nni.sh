if [ "$1" = "run" ];
then
    nnictl create --config nni/config.yaml --port 8080
fi

if [ "$1" = "stop" ];
then
    nnictl stop --all
fi

if [ "$1" = "resume" ];
then
    if [ "$2" = "" ];
    then
        echo "error: Miss Experiment ID"
    else
        nnictl resume $2
    fi
fi