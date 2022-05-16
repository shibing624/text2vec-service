FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.10
RUN pip install nlp-service
COPY ./ /app
COPY ./docker/entrypoint.sh /app
WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

