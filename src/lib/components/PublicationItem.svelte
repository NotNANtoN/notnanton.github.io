<script lang="ts">
	import Check from '~icons/lucide/check';
	import Copy from '~icons/lucide/copy';
	import ExternalLink from '~icons/lucide/external-link';
	import FileText from '~icons/lucide/file-text';
	import type { Publication } from '$lib/types';
	import { site } from '$lib/site';

	let { publication }: { publication: Publication } = $props();
	let expanded = $state(false);
	let bibtexOpen = $state(false);
	let copied = $state(false);

	const visibleAuthors = $derived(expanded ? publication.authors : publication.authors.slice(0, 3));
	const remainingAuthors = $derived(Math.max(0, publication.authors.length - 3));

	function isOwnName(author: string) {
		return new RegExp(`\\b${site.lastName}\\b`, 'i').test(author);
	}

	async function copyBibtex() {
		if (!navigator.clipboard) return;
		await navigator.clipboard.writeText(publication.bibtex);
		copied = true;
		window.setTimeout(() => (copied = false), 1800);
	}
</script>

<article class="card-surface p-5 sm:p-6">
	<div class="flex gap-4">
		<div class="hidden size-12 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-center text-xs font-bold uppercase tracking-wide text-accent sm:flex">
			{publication.abbr ?? publication.type.slice(0, 4)}
		</div>
		<div class="min-w-0 flex-1">
			<h3 class="text-lg font-semibold leading-7 tracking-tight">{publication.title}</h3>
			<div class="mt-2 text-sm leading-6 text-muted">
				{#each visibleAuthors as author, index (author + index)}
					{#if isOwnName(author)}
						<strong class="font-semibold text-ink">{author}</strong>
					{:else}
						{author}
					{/if}{index < visibleAuthors.length - 1 ? ', ' : ''}
				{/each}
				{#if !expanded && remainingAuthors > 0}
					<button
						type="button"
						class="ml-1 font-semibold text-accent hover:underline"
						aria-expanded={expanded}
						onclick={() => (expanded = true)}
					>
						+{remainingAuthors} authors
					</button>
				{:else if expanded && remainingAuthors > 0}
					<button
						type="button"
						class="ml-1 font-semibold text-accent hover:underline"
						aria-expanded={expanded}
						onclick={() => (expanded = false)}
					>
						Show fewer
					</button>
				{/if}
			</div>
			<p class="mt-2 text-sm">
				<span class="inline-flex items-center gap-1.5">
					{#if publication.venueIcon}
						<img
							class="size-4 rounded-[3px] object-contain"
							src={publication.venueIcon}
							alt=""
							aria-hidden="true"
							loading="lazy"
							decoding="async"
						/>
					{/if}
					<span>{@html publication.venueHtml}</span>
				</span>
				<span class="text-muted"> · {publication.year}</span>
			</p>

			<div class="mt-4 flex flex-wrap gap-2">
				{#each publication.links as link (link.url)}
					<a class="icon-link" href={link.url} target="_blank" rel="noreferrer">
						<ExternalLink width={14} height={14} />{link.name}
					</a>
				{/each}
				<button type="button" class="icon-link" aria-expanded={bibtexOpen} onclick={() => (bibtexOpen = !bibtexOpen)}>
					<FileText width={14} height={14} />BibTeX
				</button>
			</div>

			{#if bibtexOpen}
				<div class="relative mt-4">
					<pre class="max-h-72 overflow-auto rounded-lg border border-border bg-surface p-4 pr-12 text-xs leading-5 text-ink"><code>{publication.bibtex}</code></pre>
					<button
						type="button"
						class="absolute right-2 top-2 inline-flex size-8 items-center justify-center rounded-md text-muted hover:bg-border/50 hover:text-accent"
						aria-label={copied ? 'BibTeX copied' : 'Copy BibTeX'}
						title={copied ? 'Copied' : 'Copy BibTeX'}
						onclick={copyBibtex}
					>
						{#if copied}<Check width={16} height={16} />{:else}<Copy width={16} height={16} />{/if}
					</button>
				</div>
			{/if}
		</div>
	</div>
</article>
