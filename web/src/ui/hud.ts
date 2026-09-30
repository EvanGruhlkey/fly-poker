import type { TableView } from "../game/model";

export function hudMarkup(view: TableView): string {
  const actions = view.actions
    .map((action) => `<button class="action action-${action.kind}" data-action="${action.kind}">${action.label}</button>`)
    .join("");
  const activity = view.activity;
  return `
    <div class="scene" data-scene></div>
    <header class="topbar">
      <div class="brand" aria-label="Poker vs. a Fruit Fly">Poker vs. a <span>Fruit Fly</span></div>
      <div class="science">166,700 neurons <i></i> 124M synapses <i></i> MaleCNS v1.0</div>
      <button class="text-button" data-new-game>New game</button>
    </header>
    <section class="fly-status panel">
      <div class="eyebrow">The fly</div>
      <strong>${view.flyStackBb} BB</strong>
      <span class="thinking"><i></i> connectome online</span>
    </section>
    <section class="hand-history panel">
      <div class="eyebrow">Hand ${view.handNumber}</div>
      <div><span>Fly</span><b>raises 2.5 BB</b></div>
      <div><span>You</span><b>call</b></div>
      <div><span>Flop</span><b>A&spades; 9&diams; 4&clubs;</b></div>
      <div><span>Fly</span><b>bets 4.25 BB</b></div>
    </section>
    <section class="activity panel">
      <div class="panel-heading">
        <div><div class="eyebrow">Live connectome</div><strong>150 ms decision</strong></div>
        <button class="collapse" data-collapse aria-label="Hide connectome panel">&minus;</button>
      </div>
      <div class="brain" aria-hidden="true">
        <div class="brain-core"></div>
        <span class="pulse pulse-a"></span><span class="pulse pulse-b"></span><span class="pulse pulse-c"></span>
      </div>
      <div class="legend"><span><i class="cyan"></i>sensory in</span><span><i class="gold"></i>central</span><span><i class="red"></i>motor out</span></div>
      <div class="stats">
        <div><b>${activity.neuronsFired.toLocaleString()}</b><span>neurons fired</span></div>
        <div><b>${activity.spikes.toLocaleString()}</b><span>spikes</span></div>
        <div><b>${activity.motorSpikes.toLocaleString()}</b><span>motor spikes</span></div>
      </div>
      <p>Every candidate action runs through the fixed connectome. Only the action readout learns.</p>
    </section>
    <div class="pot-label"><span>Pot</span><strong>${view.potBb} BB</strong></div>
    <div class="player-stack">You <strong>${view.playerStackBb} BB</strong></div>
    <nav class="actions" aria-label="Poker actions">${actions}</nav>
    <div class="camera-help">Drag to orbit <i></i> Scroll to zoom <i></i> H hides the interface</div>
  `;
}
