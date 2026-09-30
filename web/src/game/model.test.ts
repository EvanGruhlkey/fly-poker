import { describe, expect, it } from "vitest";

import { createDemoTable, qualityForWidth } from "./model";

describe("createDemoTable", () => {
  it("creates a playable flop decision", () => {
    const table = createDemoTable();

    expect(table.phase).toBe("player-turn");
    expect(table.board).toEqual(["AS", "9D", "4C"]);
    expect(table.actions.map((action) => action.label)).toEqual([
      "Fold",
      "Call 4.25 BB",
      "Raise",
    ]);
    expect(table.activity.neuronsFired).toBe(12482);
  });
});

describe("qualityForWidth", () => {
  it("reduces scene cost on narrow screens", () => {
    expect(qualityForWidth(500)).toEqual({ shadows: false, pixelRatio: 1 });
    expect(qualityForWidth(1400)).toEqual({ shadows: true, pixelRatio: 1.5 });
  });
});
