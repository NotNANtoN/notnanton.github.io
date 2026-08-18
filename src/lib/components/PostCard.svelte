<script lang="ts">
	import ArrowUpRight from '~icons/lucide/arrow-up-right';
	import CalendarDays from '~icons/lucide/calendar-days';
	import Clock3 from '~icons/lucide/clock-3';
	import type { PostMeta } from '$lib/types';
	import { formatDate } from './format';
	import CardMedia from './CardMedia.svelte';
	import TagChips from './TagChips.svelte';

	let { post }: { post: PostMeta } = $props();
</script>

<article class="card-surface group flex h-full flex-col overflow-hidden transition hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-lg">
	<a href={post.url} class="block overflow-hidden no-underline" aria-label={`Read ${post.title}`}>
		{#if post.thumbnail}
			<CardMedia
				src={post.thumbnail}
				class="rounded-t-xl transition duration-300 group-hover:scale-[1.02]"
			/>
		{:else}
			<div
				class="rounded-t-xl bg-gradient-to-br from-accent/15 via-surface to-surface"
				style="aspect-ratio: 16/9;"
			></div>
		{/if}
	</a>
	<div class="flex flex-1 flex-col space-y-3 p-5">
		<div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
			<span class="inline-flex items-center gap-1"><CalendarDays width={14} height={14} />{formatDate(post.date)}</span>
			<span class="inline-flex items-center gap-1"><Clock3 width={14} height={14} />{post.readingTime}</span>
		</div>
		<h2 class="text-xl font-semibold tracking-tight">
			<a class="text-ink no-underline hover:text-accent" href={post.url}>{post.title}</a>
		</h2>
		<p class="line-clamp-3 text-sm leading-6 text-muted">{post.description}</p>
		<div class="flex-1"></div>
		<div class="flex items-end justify-between gap-4">
			<TagChips tags={post.tags} />
			<a class="inline-flex shrink-0 items-center gap-1 text-sm font-semibold no-underline" href={post.url}>
				Read <ArrowUpRight width={15} height={15} />
			</a>
		</div>
	</div>
</article>
