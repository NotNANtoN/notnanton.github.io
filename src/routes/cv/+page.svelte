<script lang="ts">
	import type { CVEntry } from '$lib/types';
	import type { PageProps } from './$types';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();

	function sectionId(title: string) {
		return title.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
	}

	function entryLabel(entry: CVEntry) {
		return entry.name ?? entry.title ?? entry.value ?? entry.description?.join(', ') ?? '';
	}

	function bullets(entry: CVEntry) {
		return entry.items ?? entry.description ?? [];
	}
</script>

<SiteHead title="CV" description="Curriculum vitae of Anton Wiehe." canonicalPath="/cv/" />

<div class="wide-container">
	<header class="mx-auto mb-10 max-w-3xl">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Background</p>
		<h1 class="text-4xl font-bold tracking-tight">Curriculum vitae</h1>
	</header>

	<div class="grid gap-10 lg:grid-cols-[12rem_minmax(0,48rem)] lg:justify-center lg:gap-12">
		{#if data.sections.length > 0}
			<aside class="lg:sticky lg:top-24 lg:self-start">
				<nav aria-label="CV sections">
					<ul class="space-y-1 border-l border-border">
						{#each data.sections as section (section.title)}
							<li>
								<a class="block border-l-2 border-transparent py-1.5 pl-3 text-sm text-muted no-underline hover:border-accent hover:text-accent" href={`#${sectionId(section.title)}`}>
									{section.title}
								</a>
							</li>
						{/each}
					</ul>
				</nav>
			</aside>
		{/if}

		<div class="min-w-0 space-y-6">
			{#each data.sections as section (section.title)}
				<section id={sectionId(section.title)} class="card-surface scroll-mt-24 p-5 sm:p-6" aria-labelledby={`${sectionId(section.title)}-heading`}>
					<h2 id={`${sectionId(section.title)}-heading`} class="mb-5 text-xl font-semibold">{section.title}</h2>

					{#if section.type === 'map'}
						<dl class="divide-y divide-border">
							{#each section.contents as entry, index (entry.name ?? entry.title ?? index)}
								<div class="grid gap-1 py-3 first:pt-0 sm:grid-cols-[10rem_1fr] sm:gap-4">
									<dt class="font-semibold">{entry.name ?? entry.title}</dt>
									<dd class="text-muted">{entry.value}</dd>
								</div>
							{/each}
						</dl>
					{:else if section.type === 'time_table'}
						<div class="space-y-6">
							{#each section.contents as entry, index (entry.title ?? entry.year ?? index)}
								<div class="relative border-l-2 border-accent/25 pl-5">
									{#if entry.year}<span class="mb-1 inline-block rounded-full bg-accent/10 px-2.5 py-1 text-xs font-bold text-accent">{entry.year}</span>{/if}
									{#if entry.title}<h3 class="font-semibold">{entry.title}</h3>{/if}
									{#if entry.institution}<p class="text-sm text-muted">{entry.institution}</p>{/if}
									{#if bullets(entry).length > 0}
										<ul class="mt-2 list-disc space-y-1 pl-5 text-sm leading-6 text-muted">
											{#each bullets(entry) as item (item)}<li>{item}</li>{/each}
										</ul>
									{/if}
								</div>
							{/each}
						</div>
					{:else if section.type === 'nested_list'}
						<div class="space-y-5">
							{#each section.contents as entry, index (entry.title ?? index)}
								<div>
									{#if entry.title}<h3 class="font-semibold">{entry.title}</h3>{/if}
									{#if bullets(entry).length > 0}
										<ul class="mt-2 list-disc space-y-1 pl-5 text-sm leading-6 text-muted">
											{#each bullets(entry) as item (item)}<li>{item}</li>{/each}
										</ul>
									{/if}
								</div>
							{/each}
						</div>
					{:else}
						<ul class="list-disc space-y-2 pl-5 text-sm leading-6 text-muted">
							{#each section.contents as entry, index (entryLabel(entry) || index)}
								<li>{entryLabel(entry)}</li>
							{/each}
						</ul>
					{/if}
				</section>
			{/each}
		</div>
	</div>
</div>
