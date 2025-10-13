Place your licensed container token here as `oscillink.lic`.

This file should contain an Ed25519-signed JWT provided by Oscillink.
The container will verify the token at startup using OSCILLINK_JWKS_URL and
export entitlements to /run/oscillink_entitlements.json and .env.
