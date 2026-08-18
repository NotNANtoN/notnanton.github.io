<script lang="ts">
	import ArrowUpRight from '~icons/lucide/arrow-up-right';
	import type { PageProps } from './$types';
	import CardMedia from '$lib/components/CardMedia.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
</script>

<SiteHead title="Projects" description="Products and tools built or co-founded by Anton Wiehe." canonicalPath="/projects/" />

<div class="wide-container">
	<header class="mx-auto mb-12 max-w-2xl text-center">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Selected work</p>
		<h1 class="text-4xl font-bold tracking-tight">Projects</h1>
		<p class="mt-3 text-lg leading-8 text-muted">Products and tools built to turn ideas about intelligence into useful systems.</p>
	</header>

	<div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
		{#each data.projects as project (project.slug)}
			{@const href = project.redirect ?? project.url}
			<article class="card-surface group relative overflow-hidden transition hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-lg">
				{#if project.img}
					<CardMedia src={project.img} alt="" ratio="16/9" class="rounded-t-xl" />
				{:else}
					<div class="h-24 bg-gradient-to-br from-accent/20 via-surface to-surface"></div>
				{/if}
				<div class="flex min-h-48 flex-col p-5">
					<p class="mb-2 text-xs font-semibold uppercase tracking-[0.15em] text-accent">{project.category}</p>
					<h2 class="text-xl font-semibold tracking-tight">
						<a
							class="text-ink no-underline after:absolute after:inset-0 group-hover:text-accent"
							href={href}
							target={project.redirect?.startsWith('http') ? '_blank' : undefined}
							rel={project.redirect?.startsWith('http') ? 'noreferrer' : undefined}
						>
							{project.title}
						</a>
					</h2>
					<p class="mt-3 flex-1 text-sm leading-6 text-muted">{project.description}</p>
					<span class="relative z-10 mt-4 inline-flex items-center gap-1 text-sm font-semibold text-accent">
						Explore <ArrowUpRight width={15} height={15} />
					</span>
				</div>
			</article>
		{:else}
			<p class="text-muted">No projects available.</p>
		{/each}
	</div>
</div>
