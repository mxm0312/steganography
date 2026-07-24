#!/usr/bin/env bash
set -e

# Без аргументов — веб-интерфейс. С аргументами — выполнить их
# (stego ..., python ..., pytest ...) в окружении с установленным пакетом.
if [ "$#" -eq 0 ]; then
    exec streamlit run stego/ui/app.py \
        --server.address=0.0.0.0 --server.port=8501
fi

exec "$@"
