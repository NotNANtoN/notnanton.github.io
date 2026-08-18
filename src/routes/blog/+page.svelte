<script lang="ts">
	import ArrowUpRight from '~icons/lucide/arrow-up-right';
	import type { PageProps } from './$types';
	import PostCard from '$lib/components/PostCard.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
</script>

<SiteHead title="Blog" description="Thoughts on AGI, cognition, and intelligence." canonicalPath="/blog/" />

<div class="page-container">
	<header class="mb-10">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Writing</p>
		<div class="flex flex-wrap items-end justify-between gap-4">
			<div>
				<h1 class="text-4xl font-bold tracking-tight">Blog</h1>
				<p class="mt-3 max-w-xl text-lg leading-7 text-muted">
					Notes on robot learning, reinforcement learning, and building adaptive systems.
				</p>
			</div>
			<a class="inline-flex items-center gap-1 text-sm font-semibold no-underline" href="/feed.xml">
				RSS feed <ArrowUpRight width={15} height={15} />
			</a>
		</div>
	</header>

	{#if data.tags.length > 0}
		<nav class="mb-10 flex flex-wrap gap-2" aria-label="Filter posts by tag">
			{#each data.tags as item (item.tag)}
				<a
					class="rounded-full border border-border px-3 py-1.5 text-sm text-muted no-underline hover:border-accent hover:text-accent"
					href={`/blog/tag/${encodeURIComponent(item.tag)}/`}
				>
					#{item.tag} <span class="text-xs opacity-70">{item.count}</span>
				</a>
			{/each}
		</nav>
	{/if}

	<div class="grid gap-6">
		{#each data.posts as post (post.url)}
			<PostCard {post} />
		{:else}
			<p class="text-muted">No posts yet.</p>
		{/each}
	</div>
</div>
