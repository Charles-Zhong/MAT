if [ "$1" = "run" ];
then
    if [ "$2" = "" ];
    then
        echo "error: Miss Experiment Name."
    else
        sed -i "1d" nni/config.yaml
        sed -i "1i experiment_name: $2" nni/config.yaml
        nnictl create --config nni/config.yaml --port 8080
    fi
fi

if [ "$1" = "stop" ];
then
    nnictl stop --all
fi

if [ "$1" = "resume" ];
then
    if [ "$2" = "" ];
    then
        echo "error: Miss Experiment ID."
    else
        nnictl resume $2
    fi
fi