<script lang="ts">
  import { onMount } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let currentTranscript: string = '';
  let lastWord: Date | null = null;

  onMount(() => {
    getSettings();
    initSpeechRecognition();
  });

  async function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error('SpeechRecognition API is not supported in this browser.');
      return;
    }

    // setInterval(() => {
    //   if (lastWord && new Date().getTime() - lastWord.getTime() > 1000) {
    //     console.log(currentTranscript);
    //     currentTranscript = '';
    //     lastWord = null;
    //   }
    // }, 1000);

    // Create a new instance of speech recognition
    var recognition = new SpeechRecognition();
    // set params
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.start();

    // Handle errors
    recognition.onerror = (event) => {
      console.error('SpeechRecognition error:', event.error);
    };

    // Start recognition
    try {
      recognition.start();
    } catch (e) {
      console.error('Error starting speech recognition:', e);
    }

    recognition.onresult = function (event) {
      // delve into words detected results & get the latest
      // total results detected
      var resultsLength = event.results.length - 1;
      // get length of latest results
      var ArrayLength = event.results[resultsLength].length - 1;
      // get last word detected
      var isFinal = event.results[resultsLength].isFinal;
      var saidWord = event.results[resultsLength][ArrayLength].transcript;
      let keyword = 'diablo';
      // console.log(isFinal, saidWord);
      if (isFinal) {
        currentTranscript = saidWord.toLowerCase();
        lastWord = new Date();
        if (currentTranscript.includes(keyword)) {
          let lastIndex = currentTranscript.lastIndexOf(keyword);
          if (lastIndex !== -1) {
            // Add the length of keyword to lastIndex to start after the keyword
            let cleanPrompt = currentTranscript.substring(lastIndex + keyword.length).trim();
            console.log(cleanPrompt);
            document.getElementById('my-favorite-textarea').value = cleanPrompt;
          } else {
            console.log('Keyword not found');
          }
        }
      }

      // // loop through links and match to word spoken
      // for (i = 0; i < allLinks.length; i++) {
      //   // get the word associated with the link
      //   var dataWord = allLinks[i].dataset.word;

      //   // if word matches chenge the colour of the link
      //   if (saidWord.indexOf(dataWord) != -1) {
      //     allLinks[i].style.color = 'red';
      //   }
      // }

      // append the last word to the bottom sentence
      // console.log(saidWord);
    };
  }

  async function getSettings() {
    const settings = await fetch('/settings').then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    console.log(pipelineParams);
    if (maxQueueSize > 0) {
      getQueueSize();
      setInterval(() => {
        getQueueSize();
      }, 2000);
    }
  }
  async function getQueueSize() {
    const data = await fetch('/queue_size').then((r) => r.json());
    currentQueueSize = data.queue_size;
  }

  function getSreamdata() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;

  let disabled = false;
  async function toggleLcmLive() {
    if (!isLCMRunning) {
      if (isImageMode) {
        await mediaStreamActions.enumerateDevices();
        await mediaStreamActions.start();
      }
      disabled = true;
      await lcmLiveActions.start(getSreamdata);
      disabled = false;
    } else {
      if (isImageMode) {
        mediaStreamActions.stop();
      }
      lcmLiveActions.stop();
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <article class="border-2 text-center">
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {#if isImageMode}
        <div class="sm:col-start-1">
          <VideoInput
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
          ></VideoInput>
        </div>
      {/if}
      <div class={isImageMode ? 'sm:col-start-2' : 'col-span-2'}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-2">
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg my-1 p-2'}>
          {#if isLCMRunning}
            Stop
          {:else}
            Start
          {/if}
        </Button>
        <PipelineOptions {pipelineParams}></PipelineOptions>
      </div>
    </article>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
