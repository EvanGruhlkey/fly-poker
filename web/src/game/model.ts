export type Card = `${
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "T"
  | "J"
  | "Q"
  | "K"
  | "A"}${"C" | "D" | "H" | "S"}`;

export type TablePhase =
  | "dealing"
  | "fly-thinking"
  | "player-turn"
  | "showdown";

export interface PlayerAction {
  readonly kind: "fold" | "call" | "raise";
  readonly label: string;
}

export interface TableView {
  readonly phase: TablePhase;
  readonly handNumber: number;
  readonly potBb: number;
  readonly playerStackBb: number;
  readonly flyStackBb: number;
  readonly playerCards: readonly [Card, Card];
  readonly board: readonly Card[];
  readonly actions: readonly PlayerAction[];
  readonly activity: {
    readonly neuronsFired: number;
    readonly spikes: number;
    readonly motorSpikes: number;
    readonly simulationMs: number;
  };
}

export interface SceneQuality {
  readonly shadows: boolean;
  readonly pixelRatio: number;
}

export function qualityForWidth(width: number): SceneQuality {
  return width < 800
    ? { shadows: false, pixelRatio: 1 }
    : { shadows: true, pixelRatio: 1.5 };
}

export function createDemoTable(): TableView {
  return {
    phase: "player-turn",
    handNumber: 18,
    potBb: 8.5,
    playerStackBb: 94,
    flyStackBb: 97.5,
    playerCards: ["KH", "QH"],
    board: ["AS", "9D", "4C"],
    actions: [
      { kind: "fold", label: "Fold" },
      { kind: "call", label: "Call 4.25 BB" },
      { kind: "raise", label: "Raise" },
    ],
    activity: {
      neuronsFired: 12482,
      spikes: 98041,
      motorSpikes: 1204,
      simulationMs: 150,
    },
  };
}
