# Security Policy

## Scope

GenRoad is a research application for local image generation. It is not
designed as a hardened multi-user service.

## Safe deployment

- Keep the default host bound to `127.0.0.1`.
- Do not use `--share` with confidential or personally identifiable images.
- If the application is exposed on a network, add authentication, TLS, rate
  limits, resource limits, and per-user storage isolation.
- Never commit `config.yaml`, credentials, model tokens, certificates, or
  generated outputs.

## Reporting

Please report security issues privately to the repository maintainers before
opening a public issue. Include reproduction steps and the affected commit.

