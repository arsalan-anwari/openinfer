#[derive(Debug, Clone, Copy)]
pub struct VulkanOpSpec<'a> {
    pub entry: &'a str,
    pub spv_dir: &'a str,
    pub workgroup_size: [u32; 3],
    pub push_constant_size: u32,
}

#[derive(Debug)]
pub enum BindingBytes<'a> {
    ReadOnly(&'a [u8]),
    ReadWrite(&'a mut [u8]),
    Alias {
        source_binding: usize,
        offset: u64,
        bytes: &'a mut [u8],
    },
}
