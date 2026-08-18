<script lang="ts">
	import Github from '~icons/simple-icons/github';
	import type { PageProps } from './$types';
	import SiteHead from '$lib/components/SiteHead.svelte';

	let { data }: PageProps = $props();
</script>

<SiteHead title="Repositories" description="Open-source projects and GitHub repositories." canonicalPath="/repositories/" />

<div class="page-container">
	<header class="mb-12">
		<p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-accent">Open source</p>
		<h1 class="text-4xl font-bold tracking-tight">Repositories</h1>
		<p class="mt-3 max-w-2xl text-lg leading-8 text-muted">Projects, experiments, and tools shared on GitHub.</p>
	</header>

	{#if data.repositories.users.length > 0}
		<section class="mb-12" aria-labelledby="github-users-heading">
			<h2 id="github-users-heading" class="mb-5 text-2xl font-semibold">GitHub users</h2>
			<div class="grid gap-5 sm:grid-cols-2">
				{#each data.repositories.users as username (username)}
					<a class="card-surface block overflow-hidden p-2 no-underline transition hover:-translate-y-0.5 hover:shadow-lg" href={`https://github.com/${username}`} target="_blank" rel="noreferrer">
						<p class="sr-only">GitHub profile for {username}</p>
						<img
							class="block w-full dark:hidden"
							src={`https://github-readme-stats.vercel.app/api/?username=${encodeURIComponent(username)}&theme=default&show_icons=true`}
							alt={username}
						/>
						<img
							class="hidden w-full dark:block"
							src={`https://github-readme-stats.vercel.app/api/?username=${encodeURIComponent(username)}&theme=dark&show_icons=true`}
							alt={username}
						/>
					</a>
				{/each}
			</div>
		</section>
	{/if}

	<section aria-labelledby="github-repositories-heading">
		<h2 id="github-repositories-heading" class="mb-5 text-2xl font-semibold">GitHub repositories</h2>
		{#if data.repositories.repos.length > 0}
			<div class="grid gap-5 sm:grid-cols-2">
				{#each data.repositories.repos as repository (repository)}
					{@const [owner, repo] = repository.split('/')}
					{@const showOwner = !data.repositories.users.includes(owner)}
					<a class="card-surface block overflow-hidden p-2 no-underline transition hover:-translate-y-0.5 hover:shadow-lg" href={`https://github.com/${repository}`} target="_blank" rel="noreferrer">
						<img
							class="block w-full dark:hidden"
							src={`https://github-readme-stats.vercel.app/api/pin/?username=${encodeURIComponent(owner)}&repo=${encodeURIComponent(repo)}&theme=default&show_owner=${showOwner}`}
							alt={repository}
						/>
						<img
							class="hidden w-full dark:block"
							src={`https://github-readme-stats.vercel.app/api/pin/?username=${encodeURIComponent(owner)}&repo=${encodeURIComponent(repo)}&theme=dark&show_owner=${showOwner}`}
							alt={repository}
						/>
					</a>
				{/each}
			</div>
		{:else}
			<p class="text-muted">No repositories listed.</p>
		{/if}
	</section>

	<a class="mt-8 inline-flex items-center gap-2 text-sm font-semibold no-underline" href="https://github.com/NotNaNtoN" target="_blank" rel="noreferrer">
		<Github width={17} height={17} /> View GitHub profile
	</a>
</div>
