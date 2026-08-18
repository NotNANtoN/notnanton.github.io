# notnanton.github.io

Personal academic website for Anton O. Wiehe, built with [SvelteKit](https://svelte.dev/docs/kit)
(Svelte 5) and hosted on [GitHub Pages](https://pages.github.com/).

**Live site:** [notnanton.github.io](https://notnanton.github.io)

## Local development

```bash
npm install
npm run dev
```

Then open [localhost:5173](http://localhost:5173).

```bash
npm run check    # svelte-check / typescript
npm run build    # static build into ./build
npm run preview  # serve the production build
```

## Content

All content lives in `content/` and is read at build time, so a rebuild is the only thing needed
after editing.

| Path                              | What it holds                                                    |
| --------------------------------- | ---------------------------------------------------------------- |
| `content/about.md`                | Home page: subtitle, profile image, intro text                   |
| `content/posts/*.md`              | Blog posts (`YYYY-MM-DD-slug.md`, URL `/blog/<year>/<slug>/`)    |
| `content/news/*.md`               | Short news items shown on the home page                          |
| `content/projects/*.md`           | Project pages and cards                                          |
| `content/bibliography/papers.bib` | Publications (BibTeX, `selected: {true}` shows on the home page) |
| `content/talks.yml`               | Talks, posters and presentations                                 |
| `content/cv.yml`                  | CV sections                                                      |
| `content/repositories.yml`        | GitHub users/repos on `/repositories/`                           |
| `static/assets/`                  | Images and video                                                 |

Post front matter supports `title`, `description`, `date`, `tags`, `authors`, `featured`,
`thumbnail`, `giscus` and `bibliography`. Citations use `<d-cite key="key"></d-cite>` and resolve
against the post's `bibliography` file; math uses `$...$` and `$$...$$`.

## Deployment

`.github/workflows/deploy.yml` builds the site on every push to `master` and publishes `build/`
to the `gh-pages` branch, which GitHub Pages serves.
