def _integrate_shirg(self):
    """Integrate SHIRG into LaViDa's vision processing pipeline"""
    
    # Store SHIRG selector and wrapper reference
    self.model.shirg_selector = self.shirg_selector
    self.model.shirg_wrapper = self  # Allow patched method to access wrapper
    
    # Store original method for reference
    if hasattr(self.model, 'encode_images'):
        self.model._original_encode_images = self.model.encode_images
        
        # Only patch if we're actually using SHIRG (alpha > 0 enables SHIRG selection)
        shirg_enabled = (self.shirg_config.get('alpha', 0) > 0)
        
        if shirg_enabled:
            # Patch encode_images to use SHIRG with correct signature
            def patched_encode_images(self, images):
                """Patched encode_images that applies SHIRG-Fovea token selection"""
                            
                # SHIRG-FOVEA-FIX: Handle LaViDa's 5-view list format
                wrapper = getattr(self, 'shirg_wrapper', None)
                vision_tower = self.get_model().get_vision_tower()
                
                if (wrapper is not None and 
                    hasattr(wrapper, '_current_question_tokens') and 
                    wrapper._current_question_tokens is not None and
                    wrapper.shirg_config.get('alpha', 0) > 0):
                    
                    try:
                        # SHIRG-Fovea: Process 5-view format (1 global + 4 peripheral)
                        if hasattr(vision_tower, 'forward_with_shirg'):
                            if wrapper.shirg_config.get('debug', False):
                                if isinstance(images, list):
                                    print(f"🔍 Using SHIRG-Fovea processing with {len(images)} views")
                                else:
                                    print(f"🔍 Using SHIRG-Fovea processing with tensor input")
                            
                            try:
                                # SHIRG-Fovea method with optional text embeddings
                                text_embeddings = wrapper._current_question_tokens
                                if text_embeddings is not None:
                                    # Validate text embeddings shape and dtype
                                    if text_embeddings.dim() != 3:
                                        text_embeddings = text_embeddings.unsqueeze(0) if text_embeddings.dim() == 2 else text_embeddings
                                    # Ensure text embeddings are on correct device
                                    if isinstance(images, list) and len(images) > 0:
                                        ref_device = images[0].device if hasattr(images[0], 'device') else None
                                        ref_dtype = images[0].dtype if hasattr(images[0], 'dtype') else None
                                    else:
                                        ref_device = images.device if hasattr(images, 'device') else None
                                        ref_dtype = images.dtype if hasattr(images, 'dtype') else None
                                        
                                    if ref_device and text_embeddings.device != ref_device:
                                        text_embeddings = text_embeddings.to(device=ref_device, dtype=ref_dtype)
                                
                                # Call SHIRG-Fovea method
                                selected_features = vision_tower.forward_with_shirg(
                                    images, text_embeddings=text_embeddings
                                )
                                
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"✅ SHIRG-Fovea processing: {selected_features.shape}")
                                
                                # Apply projector to selected features
                                image_features = self.get_model().mm_projector(selected_features)
                                return image_features
                                
                            except Exception as shirg_error:
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"❌ SHIRG forward_with_shirg failed: {shirg_error}")
                                    import traceback
                                    traceback.print_exc()
                                
                                # Fallback to baseline - process through standard vision tower
                                if wrapper.shirg_config.get('debug', False):
                                    print("📉 Falling back to baseline LaViDa processing")
                                # Use standard encode_images
                                return self._original_encode_images(images)
                        else:
                            # No SHIRG available - use baseline
                            if wrapper.shirg_config.get('debug', False):
                                print("📉 No SHIRG available, using baseline")
                            return self._original_encode_images(images)
                    except Exception as e:
                        if wrapper.shirg_config.get('debug', False):
                            print(f"⚠️ SHIRG selection failed: {e}")
                            import traceback
                            traceback.print_exc()
                        # Fallback to standard processing
                        return self._original_encode_images(images)
                else:
                    # Baseline: use standard encode_images
                    return self._original_encode_images(images)
            
            # Bind method to model instance
            import types
            self.model.encode_images = types.MethodType(patched_encode_images, self.model)
            print(f"✅ SHIRG integration enabled - encode_images method patched (alpha={self.shirg_config.get('alpha', 0)})")
        else:
            print(f"✅ SHIRG integration disabled (baseline mode, alpha={self.shirg_config.get('alpha', 0)})")
            print(f"   - Using standard LaViDa encode_images (no token selection)")
    else:
        print("⚠️ Could not find encode_images method to patch")