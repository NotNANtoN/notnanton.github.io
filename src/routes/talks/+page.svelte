<script lang="ts">
	import ChevronDown from '~icons/lucide/chevron-down';
	import MapPin from '~icons/lucide/map-pin';
	import type { PageProps } from './$types';
	import CardMedia from '$lib/components/CardMedia.svelte';
	import LinkIcon from '$lib/components/LinkIcon.svelte';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
</script>

<SiteHead title="Talks" description="Conference presentations, posters, and invited talks." canonicalPath="/talks/" />

<div class="page-container">
	<header class="mb-12">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Presentations</p>
		<h1 class="text-4xl font-bold tracking-tight">Talks</h1>
		<p class="mt-3 max-w-2xl text-lg leading-8 text-muted">Conference presentations, posters, and invited talks.</p>
	</header>

	<div class="space-y-12">
		{#each data.groups as group (group.year)}
			<section aria-labelledby={`talks-${group.year}`}>
				<h2 id={`talks-${group.year}`} class="mb-5 text-2xl font-semibold">{group.year}</h2>
				<div class="space-y-4">
					{#each group.talks as talk (talk.title + talk.date)}
						{@const mediaLink = talk.links.find((link) => link.name.toLowerCase() === 'pdf') ?? talk.links[0]}
						<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
						<article
							tabindex="0"
							class="group card-surface overflow-hidden p-3 transition hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 sm:p-4"
						>
							<div class="flex items-start gap-3">
								{#if talk.icon}
									<div class="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border border-border bg-surface">
										<img
											class="size-6 object-contain"
											src={talk.icon}
											alt=""
											aria-hidden="true"
											loading="lazy"
											decoding="async"
										/>
									</div>
								{/if}
								<div class="min-w-0 flex-1">
									<div class="flex flex-wrap items-center gap-2">
										<span class="rounded-full bg-accent/10 px-2 py-0.5 text-[0.625rem] font-bold uppercase tracking-wide text-accent">{talk.kind}</span>
										<span class="text-sm text-muted">{talk.date}</span>
									</div>
									<div class="mt-1 flex items-start gap-2">
										<h3 class="min-w-0 flex-1 text-base font-semibold leading-6 tracking-tight">{talk.title}</h3>
										<ChevronDown
											class="mt-1 size-4 shrink-0 text-muted transition-transform duration-300 group-hover:rotate-180 group-focus-within:rotate-180"
											aria-hidden="true"
										/>
									</div>
									<p class="mt-1 inline-flex items-start gap-1.5 text-sm text-muted"><MapPin class="mt-0.5 shrink-0" width={15} height={15} />{talk.venue}</p>
								</div>
							</div>

							<div
								class="grid max-h-0 overflow-hidden opacity-0 transition-[max-height,opacity] duration-300 group-hover:max-h-[1200px] group-hover:opacity-100 group-focus-within:max-h-[1200px] group-focus-within:opacity-100"
							>
								<div class="pt-3 sm:pt-4">
									{#if talk.description}
										<p class="text-sm leading-6 text-muted">{talk.description}</p>
									{/if}
									{#if talk.image}
										<div class="mt-3 w-40 shrink-0">
											{#if mediaLink}
												<a
													class="block rounded-lg no-underline focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2"
													href={mediaLink.url}
													target="_blank"
													rel="noreferrer"
													aria-label={`Open poster for ${talk.title}`}
												>
													<CardMedia
														src={talk.image}
														alt={`Poster for ${talk.title}`}
														ratio="4/3"
														class="rounded-lg border border-border"
													/>
												</a>
											{:else}
												<CardMedia
													src={talk.image}
													alt={`Poster for ${talk.title}`}
													ratio="4/3"
													class="rounded-lg border border-border"
												/>
											{/if}
										</div>
									{/if}
									{#if talk.links.length > 0}
										<div class="mt-3 flex flex-wrap gap-2">
											{#each talk.links as link (link.url)}
												<LinkIcon {link} />
											{/each}
										</div>
									{/if}
								</div>
							</div>
						</article>
					{/each}
				</div>
			</section>
		{:else}
			<p class="text-muted">No talks available.</p>
		{/each}
	</div>
</div>
