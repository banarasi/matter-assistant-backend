FROM python:3.12-slim
WORKDIR /app
# Optional corporate/AV root CAs (e.g. TLS-intercepting proxies). Drop .crt files
# into agent-service/certs/ (gitignored) to trust them during build.
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
USER appuser
EXPOSE 8080
CMD ["uvicorn", "agent_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
