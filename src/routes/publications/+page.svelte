<script lang="ts">
	import type { PageProps } from './$types';
	import PublicationItem from '$lib/components/PublicationItem.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
</script>

<SiteHead title="Publications" description="Research publications by Anton Wiehe." canonicalPath="/publications/" />

<div class="page-container">
	<header class="mb-12">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Research</p>
		<h1 class="text-4xl font-bold tracking-tight">Publications</h1>
		<p class="mt-3 max-w-2xl text-lg leading-8 text-muted">
			Selected research across medical AI, multimodal learning, and adaptive intelligence.
		</p>
	</header>

	<div class="space-y-12">
		{#each data.groups as group (group.year)}
			<section aria-labelledby={`year-${group.year}`}>
				<h2 id={`year-${group.year}`} class="mb-5 flex items-center gap-3 text-2xl font-semibold">
					<span class="h-px flex-1 bg-border"></span>
					{group.year}
					<span class="h-px flex-1 bg-border"></span>
				</h2>
				<div class="space-y-4">
					{#each group.publications as publication (publication.key)}
						<PublicationItem {publication} />
					{/each}
				</div>
			</section>
		{:else}
			<p class="text-muted">No publications available.</p>
		{/each}
	</div>
</div>
