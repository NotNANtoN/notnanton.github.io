<script lang="ts">
	import Monitor from '~icons/lucide/monitor';
	import Moon from '~icons/lucide/moon';
	import Sun from '~icons/lucide/sun';

	type ThemeMode = 'system' | 'light' | 'dark';
	const modes: ThemeMode[] = ['system', 'light', 'dark'];

	let mode = $state<ThemeMode>(
		typeof document !== 'undefined' && document.documentElement.dataset.theme
			? (document.documentElement.dataset.theme as ThemeMode)
			: 'system'
	);

	const label = $derived(
		mode === 'system' ? 'Theme: system' : mode === 'light' ? 'Theme: light' : 'Theme: dark'
	);

	function cycleTheme() {
		const nextMode = modes[(modes.indexOf(mode) + 1) % modes.length];
		mode = nextMode;
		if (typeof document === 'undefined') return;
		const isDark =
			nextMode === 'dark' ||
			(nextMode === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
		document.documentElement.classList.toggle('dark', isDark);
		document.documentElement.dataset.theme = nextMode;
		localStorage.setItem('theme', nextMode);
	}
</script>

<button
	type="button"
	class="inline-flex size-9 items-center justify-center rounded-full text-muted hover:bg-surface hover:text-accent"
	aria-label={`Change ${label.toLowerCase()}`}
	title={label}
	onclick={cycleTheme}
>
	{#if mode === 'system'}
		<Monitor width={17} height={17} stroke-width={1.8} />
	{:else if mode === 'light'}
		<Sun width={17} height={17} stroke-width={1.8} />
	{:else}
		<Moon width={17} height={17} stroke-width={1.8} />
	{/if}
</button>
