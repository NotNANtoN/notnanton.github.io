<script lang="ts">
	import { page } from '$app/state';
	import Menu from '~icons/lucide/menu';
	import X from '~icons/lucide/x';
	import { nav, site } from '$lib/site';
	import ThemeToggle from './ThemeToggle.svelte';

	let menuOpen = $state(false);

	function isActive(href: string) {
		return href === '/' ? page.url.pathname === '/' : page.url.pathname.startsWith(href);
	}
</script>

<header class="sticky top-0 z-40 border-b border-border/80 bg-page/85 backdrop-blur-md">
	<nav class="wide-container flex min-h-16 items-center justify-between gap-5" aria-label="Primary">
		<a class="shrink-0 text-base font-semibold tracking-tight text-ink no-underline" href="/">
			{site.shortName}
		</a>

		<button
			type="button"
			class="inline-flex size-10 items-center justify-center rounded-lg text-ink hover:bg-surface md:hidden"
			aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
			aria-expanded={menuOpen}
			aria-controls="site-navigation"
			onclick={() => (menuOpen = !menuOpen)}
		>
			{#if menuOpen}
				<X width={21} height={21} />
			{:else}
				<Menu width={21} height={21} />
			{/if}
		</button>

		<div
			id="site-navigation"
			class={[
				'absolute left-0 right-0 top-full border-b border-border bg-page px-4 py-3 shadow-lg md:static md:flex md:items-center md:border-0 md:bg-transparent md:p-0 md:shadow-none',
				menuOpen ? 'block' : 'hidden md:flex'
			]}
		>
			<ul class="flex flex-col gap-1 md:flex-row md:items-center md:gap-1">
				{#each nav as item (item.href)}
					<li>
						<a
							class={[
								'block rounded-md px-3 py-2 text-sm font-medium no-underline transition hover:bg-surface',
								isActive(item.href)
									? 'text-accent md:bg-transparent md:underline md:decoration-2 md:underline-offset-8'
									: 'text-muted'
							]}
							href={item.href}
							aria-current={isActive(item.href) ? 'page' : undefined}
							onclick={() => (menuOpen = false)}
						>
							{item.title}
						</a>
					</li>
				{/each}
				<li class="mt-1 border-t border-border pt-2 md:mt-0 md:border-0 md:pt-0">
					<ThemeToggle />
				</li>
			</ul>
		</div>
	</nav>
</header>
