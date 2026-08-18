<script lang="ts">
	import Hash from '~icons/lucide/hash';
	import type { PageProps } from './$types';
	import PostCard from '$lib/components/PostCard.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
</script>

<SiteHead title={`Tag: ${data.tag}`} description={`Posts tagged ${data.tag}.`} canonicalPath={`/blog/tag/${encodeURIComponent(data.tag)}/`} />

<div class="page-container">
	<header class="mb-10">
		<a class="mb-4 inline-flex items-center gap-1 text-sm font-semibold no-underline" href="/blog/">← All posts</a>
		<h1 class="flex items-center gap-2 text-4xl font-bold tracking-tight"><Hash class="text-accent" width={32} height={32} />{data.tag}</h1>
		<p class="mt-3 text-muted">An archive of posts with this tag.</p>
	</header>
	<div class="grid gap-6">
		{#each data.posts as post (post.url)}
			<PostCard {post} />
		{:else}
			<p class="text-muted">No posts found for this tag.</p>
		{/each}
	</div>
</div>
