# The hosted image: the sidecar plus LiteLLM.
#
# The Dockerfile at the repository root is deliberately dependency-free, and
# that is right for the deployment it describes -- a sidecar in front of an
# agent, with nothing to audit before it goes near a regulated system. It
# cannot run this console, though. The console's default pair is
# `nvidia_nim/...` and `meta/muse-spark-1.2`, both of which are reachable only
# through LiteLLM; the hand-rolled native clients speak OpenAI, Anthropic and
# Ollama and nothing else. The root image would start, serve the page, and fail
# the first turn with "litellm is not installed".
#
# So this is the same image with the one runtime dependency added, kept
# separate rather than folded in so the dependency-free claim stays true of the
# thing that claims it.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    AARAMSE_AUDIT_PATH=/data/gateway.jsonl \
    AARAMSE_PORT=8080

WORKDIR /app

# Copied and installed before the source so a code change does not re-resolve
# the dependency tree on every build.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

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

# No shell form: the process must receive SIGTERM directly. This matters more
# than it looks -- the server waits on an Event set by the signal handler, and
# a shell in between would swallow the signal and leave the container to be
# killed rather than shut down.
ENTRYPOINT ["python", "-m", "aaramse"]
CMD ["serve"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('AARAMSE_PORT','8080')+'/healthz', timeout=3).status==200 else 1)"
