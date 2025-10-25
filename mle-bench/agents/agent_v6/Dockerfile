FROM mlebench-env

ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

ENV PYTHONUNBUFFERED=1

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent

COPY mle-bench/agents/agent_v6/requirements.txt ${AGENT_DIR}/requirements.txt

RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

RUN conda run -n ${CONDA_ENV_NAME} pip install \
    xgboost>=2.0.0 \
    lightgbm>=4.0.0 \
    catboost>=1.2.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    scipy>=1.11.0 \
    && conda clean -afy

RUN conda run -n ${CONDA_ENV_NAME} pip install \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && conda clean -afy

RUN conda run -n ${CONDA_ENV_NAME} pip install \
    tensorflow>=2.13.0 \
    && conda clean -afy

COPY agent_v6/ ${AGENT_DIR}/agent_v6/
COPY mle-bench/agents/agent_v6/start.sh ${AGENT_DIR}/start.sh

RUN chmod +x ${AGENT_DIR}/start.sh

WORKDIR ${CODE_DIR}

