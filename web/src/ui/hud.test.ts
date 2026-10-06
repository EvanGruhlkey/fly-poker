import { describe, expect, it } from "vitest";

import { createDemoTable } from "../game/model";
import { hudMarkup } from "./hud";

describe("hudMarkup", () => {
  it("shows the hand, legal actions, and connectome activity", () => {
    const demo = createDemoTable();
    const markup = hudMarkup({
      ...demo,
      activity: { ...demo.activity, simulationMs: 173 },
    });

    expect(markup).toContain("Poker vs. a Fruit Fly");
    expect(markup).toContain("Call 4.25 BB");
    expect(markup).toContain("12,482");
    expect(markup).toContain("Live connectome");
    expect(markup).toContain('data-phase="player-turn"');
    expect(markup).toContain("173 ms decision");
  });
});
