# Launching the cambio portfolio demo

The portfolio build is a fully static, backend-free version of the execution
desk. It uses synthetic market data and keeps simulated changes in the
visitor's browser. It never sends an FX order, stores credentials, or calls the
Python API.

## Build locally

```bash
cd ui
npm ci
npm run build:portfolio
npm run preview:portfolio
```

Open `http://localhost:4173`. The deployable artifact is `ui/dist`.

## GitHub Pages

The workflow at `.github/workflows/portfolio-pages.yml` builds and publishes the
demo on pushes to `main`, or when run manually from the Actions tab.

Before the first deployment:

1. Open the repository on GitHub.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, choose **GitHub Actions**.
4. Run **Deploy portfolio demo** from the Actions tab.

For a repository named `cambio`, the default URL is:

```text
https://vitor-araujo.github.io/cambio/
```

The Vite build uses relative asset paths, so repository subpaths work without a
custom base URL.

## Netlify

| Setting | Value |
|---|---|
| Base directory | `ui` |
| Build command | `npm run build:portfolio` |
| Publish directory | `ui/dist` |

## Cloudflare Pages

| Setting | Value |
|---|---|
| Root directory | `ui` |
| Build command | `npm run build:portfolio` |
| Build output directory | `dist` |

## Vercel

| Setting | Value |
|---|---|
| Root directory | `ui` |
| Framework preset | Vite |
| Build command | `npm run build:portfolio` |
| Output directory | `dist` |

## Release check

Before publishing, verify:

- the header says **Portfolio demo** and **simulação**;
- execution and undo work after a refresh;
- controls show a local-only success message;
- Telegram says it is a simulation and accepts no credentials;
- refreshing the deployed URL returns the application;
- favicon, fonts, and chart chunks load from the deployed subpath;
- browser console and network panel contain no failed `/api` requests.

The live application remains available through `npm run dev` with
`python server.py --dev`; portfolio mode only activates through the dedicated
Vite mode.
