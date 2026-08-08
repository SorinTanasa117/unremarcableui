import { useQuery } from '@tanstack/react-query';

export function useFileSystem(watchInterval = 2000) {
  const tree = useQuery({
    queryKey: ['file-tree'],
    queryFn: async () => {
      const res = await fetch('/api/files/tree');
      if (!res.ok) throw new Error('Failed to fetch file tree');
      return res.json();
    },
    refetchInterval: watchInterval > 0 ? watchInterval : false,
  });

  const readFile = async (path: string): Promise<string> => {
    const res = await fetch(`/api/files/read?path=${encodeURIComponent(path)}`);
    if (!res.ok) throw new Error('Failed to read file');
    const data = await res.json();
    return data.content;
  };

  const writeFile = async (path: string, content: string): Promise<void> => {
    const res = await fetch('/api/files/write', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
    if (!res.ok) throw new Error('Failed to write file');
  };

  const deleteFile = async (path: string): Promise<void> => {
    const res = await fetch(`/api/files/delete?path=${encodeURIComponent(path)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete file');
    tree.refetch();
  };

  return { tree: tree.data ?? [], isLoading: tree.isLoading, readFile, writeFile, deleteFile, refetch: tree.refetch };
}
