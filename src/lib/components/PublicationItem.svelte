<script lang="ts">
	import Check from '~icons/lucide/check';
	import ChevronDown from '~icons/lucide/chevron-down';
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

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<article
	tabindex="0"
	class="group card-surface overflow-hidden p-3 transition hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 sm:p-4"
>
	<div class="flex items-start gap-3">
		<span class="mt-0.5 inline-flex min-h-6 shrink-0 items-center rounded-full bg-accent/10 px-2 text-[0.625rem] font-bold uppercase tracking-wide text-accent">
			{publication.abbr ?? publication.type.slice(0, 4)}
		</span>
		<div class="min-w-0 flex-1">
			<div class="flex items-start gap-2">
				<h3 class="min-w-0 flex-1 text-base font-semibold leading-6 tracking-tight">{publication.title}</h3>
				<ChevronDown
					class={[
						'mt-1 size-4 shrink-0 text-muted transition-transform duration-300 group-hover:rotate-180 group-focus-within:rotate-180',
						bibtexOpen && 'rotate-180'
					]}
					aria-hidden="true"
				/>
			</div>
			<p class="mt-1 text-sm text-muted">
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
				<span> · {publication.year}</span>
			</p>
		</div>
	</div>

	<div
		class={[
			'grid max-h-0 overflow-hidden opacity-0 transition-[max-height,opacity] duration-300 group-hover:max-h-[1200px] group-hover:opacity-100 group-focus-within:max-h-[1200px] group-focus-within:opacity-100',
			bibtexOpen && 'max-h-[1200px] opacity-100'
		]}
	>
		<div class="pt-3 sm:pt-4">
			<div class="text-sm leading-6 text-muted">
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

			<div class="mt-3 flex flex-wrap gap-2">
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
				<div class="relative mt-3">
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
