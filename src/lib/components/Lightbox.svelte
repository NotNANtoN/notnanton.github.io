<script lang="ts">
	let activeSrc = $state<string | null>(null);
	let activeAlt = $state('');

	function close() {
		activeSrc = null;
	}

	function open(event: MouseEvent) {
		const target = event.target;
		if (!(target instanceof HTMLImageElement) || !target.closest('figure[data-zoomable]')) return;
		activeSrc = target.currentSrc || target.src;
		activeAlt = target.alt;
	}
</script>

<svelte:window onkeydown={(event) => event.key === 'Escape' && close()} />
<svelte:document onclick={open} />

{#if activeSrc}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
		role="dialog"
		aria-modal="true"
		aria-label="Image preview"
		tabindex="-1"
		onclick={(event) => event.target === event.currentTarget && close()}
		onkeydown={(event) => event.key === 'Escape' && close()}
	>
		<button
			type="button"
			class="absolute right-4 top-4 inline-flex size-10 items-center justify-center rounded-full bg-white/10 text-2xl text-white hover:bg-white/20"
			aria-label="Close image preview"
			onclick={close}
		>
			<span aria-hidden="true">×</span>
		</button>
		<img class="max-h-[90vh] max-w-full rounded-lg object-contain" src={activeSrc} alt={activeAlt} />
	</div>
{/if}
