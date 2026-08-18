<script lang="ts">
	import ArrowUpRight from '~icons/lucide/arrow-up-right';
	import CalendarDays from '~icons/lucide/calendar-days';
	import type { PageProps } from './$types';
	import { site, socials } from '$lib/site';
	import PostCard from '$lib/components/PostCard.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';
	import SocialLinks from '$lib/components/SocialLinks.svelte';
	import { formatDate } from '$lib/components/format';

	let { data }: PageProps = $props();
</script>

<SiteHead title="About" description={site.description} canonicalPath="/" />

<div class="page-container">
	<section class="mb-16">
		<div class="mb-10 flex flex-col-reverse gap-8 sm:flex-row sm:items-start sm:justify-between">
			<header class="max-w-2xl">
				<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">About</p>
				<h1 class="text-4xl font-bold tracking-tight sm:text-5xl">{site.name}</h1>
				<p class="mt-4 text-lg leading-8 text-muted">{@html data.about.subtitleHtml}</p>
			</header>
			<div class="shrink-0 sm:pl-6">
				<img
					class={[
						'h-36 w-36 object-cover shadow-lg ring-4 ring-accent/10 sm:h-44 sm:w-44',
						data.about.profile.circular ? 'rounded-full' : 'rounded-2xl'
					]}
					src={data.about.profile.image}
					alt={site.name}
				/>
				{#if data.about.profile.moreInfoHtml}
					<div class="mt-3 text-center text-sm text-muted">{@html data.about.profile.moreInfoHtml}</div>
				{/if}
			</div>
		</div>

		<article class="prose prose-neutral max-w-none dark:prose-invert">
			{@html data.about.html}
		</article>
	</section>

	<section class="mb-16" aria-labelledby="news-heading">
		<div class="mb-5 flex items-baseline justify-between gap-4">
			<h2 id="news-heading" class="text-2xl font-semibold tracking-tight">
				<a class="text-ink no-underline hover:text-accent" href="/news/">News</a>
			</h2>
			<span class="text-sm text-muted">Recent updates</span>
		</div>
		{#if data.news.length > 0}
			<div class="max-h-72 overflow-y-auto rounded-xl border border-border bg-surface/50">
				<ul class="divide-y divide-border">
					{#each data.news as item (item.slug)}
						<li class="grid gap-2 px-4 py-3 sm:grid-cols-[7rem_1fr] sm:gap-4">
							<time class="inline-flex items-center gap-1.5 pt-0.5 text-xs font-medium text-muted" datetime={item.date}>
								<CalendarDays width={14} height={14} />{formatDate(item.date)}
							</time>
							{#if item.inline}
								<div class="text-sm leading-6 text-ink">{@html item.html}</div>
							{:else}
								<a class="font-medium no-underline" href={item.url}>{item.title}<ArrowUpRight class="ml-1 inline" width={14} height={14} /></a>
							{/if}
						</li>
					{/each}
				</ul>
			</div>
		{:else}
			<p class="text-muted">No news so far.</p>
		{/if}
	</section>

	<section class="mb-16" aria-labelledby="latest-posts-heading">
		<div class="mb-5 flex items-baseline justify-between gap-4">
			<h2 id="latest-posts-heading" class="text-2xl font-semibold tracking-tight">
				<a class="text-ink no-underline hover:text-accent" href="/blog/">Latest posts</a>
			</h2>
			<a class="text-sm font-semibold no-underline" href="/blog/">View all <ArrowUpRight class="inline" width={14} height={14} /></a>
		</div>
		<div class="grid gap-5 md:grid-cols-3">
			{#each data.posts as post (post.url)}
				<PostCard {post} />
			{/each}
		</div>
	</section>

	<section class="mb-16" aria-labelledby="selected-papers-heading">
		<div class="mb-5 flex items-baseline justify-between gap-4">
			<h2 id="selected-papers-heading" class="text-2xl font-semibold tracking-tight">
				<a class="text-ink no-underline hover:text-accent" href="/publications/">Selected publications</a>
			</h2>
			<a class="text-sm font-semibold no-underline" href="/publications/">View all <ArrowUpRight class="inline" width={14} height={14} /></a>
		</div>
		<div class="space-y-4">
			{#each data.selectedPublications as publication (publication.key)}
				<article class="border-l-2 border-accent/40 pl-4">
					<h3 class="font-semibold leading-6">{publication.title}</h3>
					<p class="mt-1 text-sm text-muted">{publication.authors.join(', ')}</p>
					<p class="mt-1 text-sm">{@html publication.venueHtml} <span class="text-muted">({publication.year})</span></p>
				</article>
			{/each}
		</div>
	</section>

	<section class="border-t border-border pt-8" aria-label="Social links">
		<p class="mb-4 text-sm text-muted">{site.contactNote}</p>
		<SocialLinks links={socials} />
	</section>
</div>
