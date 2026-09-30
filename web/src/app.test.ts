import { describe, expect, it } from "vitest";

import { createAppTitle } from "./app";

describe("createAppTitle", () => {
  it("names the game for the player", () => {
    expect(createAppTitle()).toBe("Poker vs. a Fruit Fly");
  });
});
