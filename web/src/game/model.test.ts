import { describe,expect,it } from 'vitest';
import { qualityForWidth } from './model';
describe('scene quality',()=>{
 it('reduces rendering cost on narrow screens',()=>{
  expect(qualityForWidth(390)).toEqual({shadows:false,pixelRatio:1});
  expect(qualityForWidth(1280)).toEqual({shadows:true,pixelRatio:1.5});
 });
});
