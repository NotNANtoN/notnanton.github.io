<script lang="ts">
	import CalendarDays from '~icons/lucide/calendar-days';
	import type { PageProps } from './$types';
	import SiteHead from '$lib/components/SiteHead.svelte';
	import { formatDate } from '$lib/components/format';

	let { data }: PageProps = $props();
</script>

<SiteHead title="News" description="Recent updates from Anton Wiehe." canonicalPath="/news/" />

<div class="page-container">
	<header class="mb-10">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Updates</p>
		<h1 class="text-4xl font-bold tracking-tight">News</h1>
	</header>
	<div class="space-y-4">
		{#each data.items as item (item.slug)}
			<article class="card-surface p-5">
				<time class="inline-flex items-center gap-1.5 text-xs font-medium text-muted" datetime={item.date}><CalendarDays width={14} height={14} />{formatDate(item.date)}</time>
				<h2 class="mt-2 text-xl font-semibold">
					{#if item.inline}
						{item.title}
					{:else}
						<a class="text-ink no-underline hover:text-accent" href={item.url}>{item.title}</a>
					{/if}
				</h2>
				<div class="prose prose-sm prose-neutral mt-3 max-w-none dark:prose-invert">{@html item.html}</div>
			</article>
		{:else}
			<p class="text-muted">No news so far.</p>
		{/each}
	</div>
</div>
