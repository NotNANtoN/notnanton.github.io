<script lang="ts">
	import ArrowUpRight from '~icons/lucide/arrow-up-right';
	import type { PageProps } from './$types';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
	const isExternal = $derived(data.project.redirect?.startsWith('http') ?? false);
</script>

<SiteHead title={data.project.title} description={data.project.description} canonicalPath={data.project.url} />

<svelte:head>
	{#if data.project.redirect}
		<meta http-equiv="refresh" content={`0; url=${data.project.redirect}`} />
	{/if}
</svelte:head>

{#if data.project.redirect}
	<div class="page-container text-center">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Project</p>
		<h1 class="text-4xl font-bold tracking-tight">{data.project.title}</h1>
		<p class="mt-4 text-muted">Redirecting…</p>
		<a
			class="mt-6 inline-flex items-center gap-1 font-semibold"
			href={data.project.redirect}
			target={isExternal ? '_blank' : undefined}
			rel={isExternal ? 'noreferrer' : undefined}
		>
			Continue to project <ArrowUpRight width={16} height={16} />
		</a>
	</div>
{:else}
	<div class="page-container">
		<a class="mb-8 inline-flex items-center gap-1 text-sm font-semibold no-underline" href="/projects/">← All projects</a>
		<header class="mb-10">
			<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">{data.project.category}</p>
			<h1 class="text-4xl font-bold tracking-tight sm:text-5xl">{data.project.title}</h1>
			<p class="mt-4 text-lg leading-8 text-muted">{data.project.description}</p>
		</header>
		<article class="prose prose-neutral max-w-none dark:prose-invert">
			{@html data.project.html}
		</article>
	</div>
{/if}
