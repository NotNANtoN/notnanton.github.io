<script lang="ts">
	import ArrowLeft from '~icons/lucide/arrow-left';
	import ArrowRight from '~icons/lucide/arrow-right';
	import CalendarDays from '~icons/lucide/calendar-days';
	import Clock3 from '~icons/lucide/clock-3';
	import type { PageProps } from './$types';
	import Giscus from '$lib/components/Giscus.svelte';
	import Lightbox from '$lib/components/Lightbox.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';
	import TagChips from '$lib/components/TagChips.svelte';
	import { formatDate } from '$lib/components/format';

	let { data }: PageProps = $props();
	let activeId = $state<string | null>(null);
	const firstTocId = $derived(data.post.toc[0]?.id ?? null);
	const currentId = $derived(activeId ?? firstTocId);

	$effect(() => {
		const headings = data.post.toc
			.map((item) => document.getElementById(item.id))
			.filter((heading): heading is HTMLElement => heading !== null);
		if (headings.length === 0) return;

		const observer = new IntersectionObserver(
			(entries) => {
				const visible = entries
					.filter((entry) => entry.isIntersecting)
					.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
				if (visible[0]) activeId = visible[0].target.id;
			},
			{ rootMargin: '-18% 0px -68% 0px', threshold: [0, 1] }
		);
		headings.forEach((heading) => observer.observe(heading));
		return () => observer.disconnect();
	});
</script>

<SiteHead title={data.post.title} description={data.post.description} canonicalPath={data.post.url} />

<div class="wide-container">
	<header class="mx-auto mb-10 max-w-3xl">
		<a class="mb-5 inline-flex items-center gap-1 text-sm font-semibold no-underline" href="/blog/">← Back to blog</a>
		<h1 class="text-4xl font-bold leading-tight tracking-tight sm:text-5xl">{data.post.title}</h1>
		<p class="mt-4 max-w-2xl text-lg leading-8 text-muted">{data.post.description}</p>
		<div class="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-muted">
			<span class="inline-flex items-center gap-1.5"><CalendarDays width={16} height={16} />{formatDate(data.post.date)}</span>
			<span class="inline-flex items-center gap-1.5"><Clock3 width={16} height={16} />{data.post.readingTime}</span>
		</div>
		{#if data.post.authors.length > 0}
			<div class="mt-5 flex flex-wrap gap-x-5 gap-y-3 border-t border-border pt-4">
				{#each data.post.authors as author (author.name)}
					<div class="text-sm">
						{#if author.url}
							<a class="font-semibold no-underline" href={author.url} target="_blank" rel="noreferrer">{author.name}</a>
						{:else}
							<span class="font-semibold">{author.name}</span>
						{/if}
						{#if author.affiliation}
							<span class="block text-xs text-muted">{author.affiliation}</span>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
		<div class="mt-5"><TagChips tags={data.post.tags} /></div>
	</header>

	<div class="grid gap-10 lg:grid-cols-[13rem_minmax(0,48rem)] lg:justify-center lg:gap-12">
		{#if data.post.toc.length > 0}
			<aside class="lg:sticky lg:top-24 lg:self-start" aria-label="Table of contents">
				<p class="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-muted">Contents</p>
				<nav>
					<ul class="space-y-1 border-l border-border">
						{#each data.post.toc as item (item.id)}
							<li>
								<a
									class={[
										'block border-l-2 py-1.5 text-sm leading-5 no-underline transition',
										item.depth === 3 ? 'pl-5' : 'pl-3',
										currentId === item.id
											? '-ml-px border-accent font-semibold text-accent'
											: 'border-transparent text-muted hover:text-accent'
									]}
									href={`#${item.id}`}
									aria-current={currentId === item.id ? 'location' : undefined}
								>
									{item.title}
								</a>
							</li>
						{/each}
					</ul>
				</nav>
			</aside>
		{/if}

		<div class="min-w-0">
			<article class="prose prose-neutral max-w-none dark:prose-invert">
				{@html data.post.html}
			</article>

			{#if data.post.references.length > 0}
				<section class="mt-14 border-t border-border pt-8" aria-labelledby="references-heading">
					<h2 id="references-heading" class="text-2xl font-semibold">References</h2>
					<ol class="mt-5 list-decimal space-y-4 pl-6 text-sm leading-6 text-muted marker:text-muted">
						{#each data.post.references as reference (reference.key)}
							<li id={`ref-${reference.key}`} class="scroll-mt-24 pl-2">
								{#if reference.url}
									<a href={reference.url} target="_blank" rel="noreferrer">{@html reference.html}</a>
								{:else}
									{@html reference.html}
								{/if}
							</li>
						{/each}
					</ol>
				</section>
			{/if}

			<div class="mt-12 flex flex-col gap-4 border-t border-border pt-6 sm:flex-row sm:justify-between">
				{#if data.previous}
					<a class="group max-w-xs no-underline" href={data.previous.url}>
						<span class="mb-1 flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted"><ArrowLeft width={14} height={14} />Older</span>
						<span class="font-semibold text-ink group-hover:text-accent">{data.previous.title}</span>
					</a>
				{:else}
					<span></span>
				{/if}
				{#if data.next}
					<a class="group max-w-xs text-right no-underline" href={data.next.url}>
						<span class="mb-1 flex items-center justify-end gap-1 text-xs font-semibold uppercase tracking-wide text-muted">Newer<ArrowRight width={14} height={14} /></span>
						<span class="font-semibold text-ink group-hover:text-accent">{data.next.title}</span>
					</a>
				{/if}
			</div>

			<Giscus enabled={data.post.giscus} title={data.post.title} />
		</div>
	</div>
</div>

<Lightbox />
