# AARAMSE sidecar.
#
# Deliberately dependency-free: the package imports nothing outside the standard
# library, so the image is a stock Python base plus the source. Nothing is
# fetched at build time, which means the build is reproducible offline and there
# is no dependency tree to audit before putting this in front of a regulated
# agent.
FROM python:3.13-slim

# Fail fast and log unbuffered: a sidecar's stdout is its operational record.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    AARAMSE_AUDIT_PATH=/data/gateway.jsonl \
    AARAMSE_PORT=8080

WORKDIR /app
COPY src/ /app/src/
COPY data/ /app/data/

# The audit log is the artifact a supervisor reads; keep it on a volume so a
# container restart cannot silently discard the chain.
RUN mkdir -p /data && \
    useradd --system --uid 10001 aaramse && \
    chown -R aaramse:aaramse /data /app
VOLUME ["/data"]
USER aaramse

EXPOSE 8080

# No shell form: the process must receive SIGTERM directly so the server can
# shut down and the audit file is closed cleanly.
ENTRYPOINT ["python", "-m", "aaramse"]
CMD ["serve"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('AARAMSE_PORT','8080')+'/healthz', timeout=3).status==200 else 1)"
